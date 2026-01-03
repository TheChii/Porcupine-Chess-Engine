//! NNUE wrapper for Aurora NNUE with incremental update support.
//!
//! Uses ferrum-nnue with Aurora architecture (768→256×2→1) for fast evaluation.

use crate::types::{Board, Score, ToNnue, Move, Piece, Color, MoveFlag};
use nnue::aurora::{load_model as aurora_load, AuroraModel, AuroraState};
use std::sync::Arc;
use movegen::Square;

/// Global type for shared thread-safe model
pub type Model = Arc<AuroraModel>;

/// Load NNUE model from file
pub fn load_model(path: &str) -> std::io::Result<Model> {
    match aurora_load(path) {
        Ok(model) => Ok(Arc::new(model)),
        Err(e) => Err(e),
    }
}

/// Create a fresh NNUE state from a board position
pub fn create_state<'m>(model: &'m AuroraModel, board: &Board) -> AuroraState<'m> {
    let mut state = model.new_state();

    // Add ALL pieces (including kings - Aurora treats kings as regular features)
    for &piece in &[Piece::Pawn, Piece::Knight, Piece::Bishop, 
                    Piece::Rook, Piece::Queen, Piece::King] {
        for &color in &[Color::White, Color::Black] {
            let bb = board.piece_bb(piece) & board.color_bb(color);
            let nnue_piece = piece.to_nnue();
            let nnue_color = color.to_nnue();
            
            for sq in bb {
                let nnue_sq = sq.to_nnue();
                // Aurora: single add() handles both perspectives
                state.add(nnue_piece, nnue_color, nnue_sq);
            }
        }
    }
    
    state
}

/// Evaluate using a pre-built state (fast - just runs network)
#[inline]
pub fn evaluate_state(state: &AuroraState<'_>, side_to_move: Color) -> Score {
    // Aurora's activate() returns centipawns directly
    let cp = state.activate(side_to_move.to_nnue());
    Score::cp(cp)
}

/// Evaluate from scratch (creates new state)
#[inline]
pub fn evaluate_scratch(model: &AuroraModel, board: &Board) -> Score {
    let state = create_state(model, board);
    evaluate_state(&state, board.turn())
}

/// Update state for a move (incremental)
/// Returns true if update succeeded, false if full refresh needed
#[inline]
pub fn update_state_for_move(
    state: &mut AuroraState<'_>,
    board: &Board,  // Position BEFORE the move
    mv: Move,
) -> bool {
    let from = mv.from();
    let to = mv.to();
    let (moving_piece, moving_color) = match board.piece_at(from) {
        Some((p, c)) => (p, c),
        None => return false,
    };
    let captured = board.piece_at(to).map(|(p, _)| p);

    let nnue_piece = moving_piece.to_nnue();
    let nnue_color = moving_color.to_nnue();
    let from_sq = from.to_nnue();
    let to_sq = to.to_nnue();

    // Remove piece from old square (Aurora: single call handles both perspectives)
    state.sub(nnue_piece, nnue_color, from_sq);

    // Handle capture
    if let Some(captured_piece) = captured {
        let cap_nnue = captured_piece.to_nnue();
        let cap_color = (!moving_color).to_nnue();
        state.sub(cap_nnue, cap_color, to_sq);
    }

    // Handle en passant capture
    if moving_piece == Piece::Pawn && mv.flag() == MoveFlag::EnPassant {
        // Remove en passant captured pawn
        let ep_sq = if moving_color == Color::White {
            Square::from_file_rank(to.file(), movegen::Rank::R5).to_nnue()
        } else {
            Square::from_file_rank(to.file(), movegen::Rank::R4).to_nnue()
        };
        let cap_color = (!moving_color).to_nnue();
        state.sub(nnue::Piece::Pawn, cap_color, ep_sq);
    }

    // Handle promotion
    let final_piece = if let Some(promo) = mv.flag().promotion_piece() {
        promo.to_nnue()
    } else {
        nnue_piece
    };

    // Add piece to new square
    state.add(final_piece, nnue_color, to_sq);

    // Handle castling: rook also moves
    let mv_flag = mv.flag();
    let is_castling = mv_flag == MoveFlag::KingCastle || mv_flag == MoveFlag::QueenCastle;
    
    if is_castling {
        let nnue_rook_color = moving_color.to_nnue();
        let (rook_from, rook_to) = if mv_flag == MoveFlag::KingCastle {
            // King-side castling
            let rank = from.rank();
            (
                Square::from_file_rank(movegen::File::H, rank),
                Square::from_file_rank(movegen::File::F, rank)
            )
        } else {
            // Queen-side castling
            let rank = from.rank();
            (
                Square::from_file_rank(movegen::File::A, rank),
                Square::from_file_rank(movegen::File::D, rank)
            )
        };
        
        let rook_from_nnue = rook_from.to_nnue();
        let rook_to_nnue = rook_to.to_nnue();
        
        state.sub(nnue::Piece::Rook, nnue_rook_color, rook_from_nnue);
        state.add(nnue::Piece::Rook, nnue_rook_color, rook_to_nnue);
    }

    true
}

/// Refresh state completely from a board position
#[inline]
pub fn refresh_state<'m>(state: &mut AuroraState<'m>, model: &'m AuroraModel, board: &Board) {
    // Create a new state and copy it
    *state = create_state(model, board);
}

/// Stateful NNUE evaluator for use in search
/// Manages a cloneable state for efficient incremental updates
pub struct NnueEvaluator<'m> {
    model: &'m AuroraModel,
    state: AuroraState<'m>,
}

impl<'m> NnueEvaluator<'m> {
    /// Create a new evaluator for a position
    pub fn new(model: &'m AuroraModel, board: &Board) -> Self {
        Self {
            model,
            state: create_state(model, board),
        }
    }

    /// Evaluate current position
    #[inline]
    pub fn evaluate(&self, side_to_move: Color) -> Score {
        evaluate_state(&self.state, side_to_move)
    }

    /// Update for a move, returns false if refresh needed
    #[inline]
    pub fn update_move(&mut self, board: &Board, mv: Move) -> bool {
        update_state_for_move(&mut self.state, board, mv)
    }

    /// Refresh state for a new position
    #[inline]
    pub fn refresh(&mut self, board: &Board) {
        self.state = create_state(self.model, board);
    }

    /// Clone the current state (for search recursion)
    #[inline]
    pub fn clone_state(&self) -> AuroraState<'m> {
        self.state.clone()
    }

    /// Restore state from a clone
    #[inline]
    pub fn restore_state(&mut self, state: AuroraState<'m>) {
        self.state = state;
    }
}

impl<'m> Clone for NnueEvaluator<'m> {
    fn clone(&self) -> Self {
        Self {
            model: self.model,
            state: self.state.clone(),
        }
    }
}
