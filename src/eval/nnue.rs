//! NNUE wrapper for HalfKP NNUE with incremental update support.
//!
//! Uses ferrum-nnue with Stockfish HalfKP architecture (40960→256×2→32→32→1).

use crate::types::{Board, Color, Move, MoveFlag, Piece, Score, ToNnue};
use binread::BinRead;
use movegen::Square;
use nnue::stockfish::halfkp::{
    scale_nn_to_centipawns, SfHalfKpFullModel, SfHalfKpModel, SfHalfKpState,
};
use std::fs::File;
use std::io::{BufReader, Cursor};
use std::sync::Arc;

/// Embedded NNUE network file (compiled into the binary)
const EMBEDDED_NNUE: &[u8] = include_bytes!("../../network.nnue");

/// Global type for shared thread-safe model
pub type Model = Arc<SfHalfKpModel>;

/// Load NNUE model from embedded bytes (no external file needed)
pub fn load_embedded_model() -> std::io::Result<Model> {
    let mut cursor = Cursor::new(EMBEDDED_NNUE);

    match SfHalfKpFullModel::read(&mut cursor) {
        Ok(full_model) => Ok(Arc::new(full_model.model)),
        Err(e) => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to parse embedded NNUE: {:?}", e),
        )),
    }
}

/// Load NNUE model from file (for custom networks)
pub fn load_model(path: &str) -> std::io::Result<Model> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    match SfHalfKpFullModel::read(&mut reader) {
        Ok(full_model) => Ok(Arc::new(full_model.model)),
        Err(e) => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to parse NNUE file: {:?}", e),
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
    for &piece in &[
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
    ] {
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

fn find_king_square(b: &Board, c: Color) -> Square {
    (b.piece_bb(Piece::King) & b.color_bb(c)).into_iter().next().unwrap_or(Square::E1)
}

#[inline]
pub fn evaluate_state(s: &mut SfHalfKpState<'_>, stm: Color) -> Score {
    let raw = s.activate(stm.to_nnue());
    Score::cp(scale_nn_to_centipawns(raw[0]))
}

#[inline]
pub fn evaluate_scratch(m: &SfHalfKpModel, b: &Board) -> Score {
    evaluate_state(&mut create_state(m, b), b.turn())
}

/// Update state for a move (incremental)
/// Returns true if update succeeded, false if full refresh needed
#[inline]
pub fn update_state_for_move(
    state: &mut SfHalfKpState<'_>,
    board: &Board, // Position BEFORE the move
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
                Square::from_file_rank(movegen::File::F, rank),
            )
        } else {
            // Queen-side castling
            let rank = from.rank();
            (
                Square::from_file_rank(movegen::File::A, rank),
                Square::from_file_rank(movegen::File::D, rank),
            )
        };

        let rook_from_nnue = rook_from.to_nnue();
        let rook_to_nnue = rook_to.to_nnue();

        state.sub(
            nnue::Color::White,
            nnue::Piece::Rook,
            nnue_rook_color,
            rook_from_nnue,
        );
        state.sub(
            nnue::Color::Black,
            nnue::Piece::Rook,
            nnue_rook_color,
            rook_from_nnue,
        );
        state.add(
            nnue::Color::White,
            nnue::Piece::Rook,
            nnue_rook_color,
            rook_to_nnue,
        );
        state.add(
            nnue::Color::Black,
            nnue::Piece::Rook,
            nnue_rook_color,
            rook_to_nnue,
        );
    }

    true
}

#[inline]
pub fn refresh_state<'m>(s: &mut SfHalfKpState<'m>, m: &'m SfHalfKpModel, b: &Board) {
    *s = create_state(m, b);
}

const MAX_PLY: usize = 128;

pub struct NnueEvaluator<'m> {
    model: &'m SfHalfKpModel,
    states: Vec<SfHalfKpState<'m>>,
}

impl<'m> NnueEvaluator<'m> {
    pub fn new(m: &'m SfHalfKpModel, b: &Board) -> Self {
        Self { model: m, states: vec![create_state(m, b); MAX_PLY] }
    }

    #[inline]
    pub fn evaluate(&mut self, p: usize, stm: Color) -> Score {
        let max_idx = self.states.len() - 1;
        let idx = p.min(max_idx);
        evaluate_state(&mut self.states[idx], stm)
    }

    #[inline]
    pub fn update_move(&mut self, p: usize, b: &Board, m: Move) -> bool {
        let np = p + 1;
        if np >= self.states.len() { self.states.push(self.states.last().unwrap().clone()); }
        let max_idx = self.states.len().saturating_sub(2);
        let src_idx = p.min(max_idx);
        self.states[np] = self.states[src_idx].clone();
        update_state_for_move(&mut self.states[np], b, m)
    }

    #[inline]
    pub fn refresh(&mut self, p: usize, b: &Board) {
        if p >= self.states.len() { self.states.resize(p + 1, self.states.last().unwrap().clone()); }
        self.states[p] = create_state(self.model, b);
    }
}

impl<'m> Clone for NnueEvaluator<'m> {
    fn clone(&self) -> Self {
        Self {
            model: self.model,
            states: self.states.clone(),
        }
    }
}
