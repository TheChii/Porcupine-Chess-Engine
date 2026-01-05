//! NNUE wrapper for HalfKP NNUE with incremental update support.
//!
//! Uses ferrum-nnue with Stockfish HalfKP architecture (40960→256×2→32→32→1).

use crate::types::{Board, Score, ToNnue, Move, Piece, Color, MoveFlag};
use nnue::stockfish::halfkp::{SfHalfKpFullModel, SfHalfKpModel, SfHalfKpState, scale_nn_to_centipawns};
use binread::BinRead;
use std::sync::Arc;
use std::fs::File;
use std::io::BufReader;
use movegen::Square;

/// Global type for shared thread-safe model
pub type Model = Arc<SfHalfKpModel>;

/// Load NNUE model from file
pub fn load_model(path: &str) -> std::io::Result<Model> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    
    match SfHalfKpFullModel::read(&mut reader) {
        Ok(full_model) => Ok(Arc::new(full_model.model)),
        Err(e) => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to parse NNUE file: {:?}", e)
        )),
    }
}

/// Create a fresh NNUE state from a board position
pub fn create_state<'m>(model: &'m SfHalfKpModel, board: &Board) -> SfHalfKpState<'m> {
    // Find king positions
    let white_king_sq = find_king_square(board, Color::White);
    let black_king_sq = find_king_square(board, Color::Black);
    
    let mut state = model.new_state(white_king_sq.to_nnue(), black_king_sq.to_nnue());

    // Add all NON-KING pieces (HalfKP does not include kings as features)
    for &piece in &[Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen] {
        for &color in &[Color::White, Color::Black] {
            let bb = board.piece_bb(piece) & board.color_bb(color);
            let nnue_piece = piece.to_nnue();
            let nnue_color = color.to_nnue();
            
            for sq in bb {
                let nnue_sq = sq.to_nnue();
                // HalfKP: add to BOTH perspectives
                state.add(nnue::Color::White, nnue_piece, nnue_color, nnue_sq);
                state.add(nnue::Color::Black, nnue_piece, nnue_color, nnue_sq);
            }
        }
    }
    
    state
}

/// Find king square for a color
fn find_king_square(board: &Board, color: Color) -> Square {
    let king_bb = board.piece_bb(Piece::King) & board.color_bb(color);
    // There should always be exactly one king
    for sq in king_bb {
        return sq;
    }
    // Fallback (should never happen in valid position)
    Square::E1
}

/// Evaluate using a pre-built state (fast - just runs network)
#[inline]
pub fn evaluate_state(state: &mut SfHalfKpState<'_>, side_to_move: Color) -> Score {
    let raw = state.activate(side_to_move.to_nnue());
    let cp = scale_nn_to_centipawns(raw[0]);
    Score::cp(cp)
}

/// Evaluate from scratch (creates new state)
#[inline]
pub fn evaluate_scratch(model: &SfHalfKpModel, board: &Board) -> Score {
    let mut state = create_state(model, board);
    evaluate_state(&mut state, board.turn())
}

/// Update state for a move (incremental)
/// Returns true if update succeeded, false if full refresh needed
#[inline]
pub fn update_state_for_move(
    state: &mut SfHalfKpState<'_>,
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

    // If king moves, we need full refresh (king position changes all feature indices)
    if moving_piece == Piece::King {
        return false; // Signal caller to do full refresh
    }

    let nnue_piece = moving_piece.to_nnue();
    let nnue_color = moving_color.to_nnue();
    let from_sq = from.to_nnue();
    let to_sq = to.to_nnue();

    // Remove piece from old square (both perspectives)
    state.sub(nnue::Color::White, nnue_piece, nnue_color, from_sq);
    state.sub(nnue::Color::Black, nnue_piece, nnue_color, from_sq);

    // Handle capture (not kings - can't capture kings)
    if let Some(captured_piece) = captured {
        if captured_piece != Piece::King {
            let cap_nnue = captured_piece.to_nnue();
            let cap_color = (!moving_color).to_nnue();
            state.sub(nnue::Color::White, cap_nnue, cap_color, to_sq);
            state.sub(nnue::Color::Black, cap_nnue, cap_color, to_sq);
        }
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
        state.sub(nnue::Color::White, nnue::Piece::Pawn, cap_color, ep_sq);
        state.sub(nnue::Color::Black, nnue::Piece::Pawn, cap_color, ep_sq);
    }

    // Handle promotion
    let final_piece = if let Some(promo) = mv.flag().promotion_piece() {
        promo.to_nnue()
    } else {
        nnue_piece
    };

    // Add piece to new square (both perspectives)
    state.add(nnue::Color::White, final_piece, nnue_color, to_sq);
    state.add(nnue::Color::Black, final_piece, nnue_color, to_sq);

    // Handle castling: rook also moves (king move was handled above with full refresh)
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
        
        state.sub(nnue::Color::White, nnue::Piece::Rook, nnue_rook_color, rook_from_nnue);
        state.sub(nnue::Color::Black, nnue::Piece::Rook, nnue_rook_color, rook_from_nnue);
        state.add(nnue::Color::White, nnue::Piece::Rook, nnue_rook_color, rook_to_nnue);
        state.add(nnue::Color::Black, nnue::Piece::Rook, nnue_rook_color, rook_to_nnue);
    }

    true
}

/// Refresh state completely from a board position
#[inline]
pub fn refresh_state<'m>(state: &mut SfHalfKpState<'m>, model: &'m SfHalfKpModel, board: &Board) {
    // Create a new state and copy it
    *state = create_state(model, board);
}

/// Stateful NNUE evaluator for use in search
/// Manages a cloneable state for efficient incremental updates
pub struct NnueEvaluator<'m> {
    model: &'m SfHalfKpModel,
    state: SfHalfKpState<'m>,
}

impl<'m> NnueEvaluator<'m> {
    /// Create a new evaluator for a position
    pub fn new(model: &'m SfHalfKpModel, board: &Board) -> Self {
        Self {
            model,
            state: create_state(model, board),
        }
    }

    /// Evaluate current position
    #[inline]
    pub fn evaluate(&mut self, side_to_move: Color) -> Score {
        evaluate_state(&mut self.state, side_to_move)
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
    pub fn clone_state(&self) -> SfHalfKpState<'m> {
        self.state.clone()
    }

    /// Restore state from a clone
    #[inline]
    pub fn restore_state(&mut self, state: SfHalfKpState<'m>) {
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
