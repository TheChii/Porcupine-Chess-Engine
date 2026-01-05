//! UCI (Universal Chess Interface) protocol handler.
//!
//! This module implements the UCI protocol for communication with chess GUIs.
//! See: http://wbec-ridderkerk.nl/html/UCIProtocol.html

mod parser;
mod handler;

pub use handler::UciHandler;

use crate::types::{Board, Move, Depth, Piece};
use movegen::Square;

/// UCI engine identification
pub const ENGINE_NAME: &str = "Porcupine";
pub const ENGINE_AUTHOR: &str = "Chiriac Theodor";

/// Time control parameters from "go" command
#[derive(Debug, Clone, Default)]
pub struct SearchParams {
    /// Search to this depth
    pub depth: Option<Depth>,
    /// Search for this many milliseconds
    pub movetime: Option<u64>,
    /// White time remaining (ms)
    pub wtime: Option<u64>,
    /// Black time remaining (ms)
    pub btime: Option<u64>,
    /// White increment per move (ms)
    pub winc: Option<u64>,
    /// Black increment per move (ms)
    pub binc: Option<u64>,
    /// Moves until next time control
    pub movestogo: Option<u32>,
    /// Infinite search (until "stop")
    pub infinite: bool,
    /// Ponder mode
    pub ponder: bool,
    /// Only search these moves
    pub searchmoves: Vec<Move>,
    /// Search for mate in N moves
    pub mate: Option<u32>,
    /// Maximum nodes to search
    pub nodes: Option<u64>,
}

impl SearchParams {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create params for a fixed depth search
    pub fn fixed_depth(depth: i32) -> Self {
        Self {
            depth: Some(Depth::new(depth)),
            ..Default::default()
        }
    }

    /// Create params for a fixed time search
    pub fn fixed_time(ms: u64) -> Self {
        Self {
            movetime: Some(ms),
            ..Default::default()
        }
    }
}

/// Parse a move string (e.g., "e2e4", "e7e8q") into a Move for the given board
pub fn parse_move(board: &Board, move_str: &str) -> Option<Move> {
    let move_str = move_str.trim().to_lowercase();
    if move_str.len() < 4 {
        return None;
    }

    // Parse source and destination squares
    let from_str = &move_str[0..2];
    let to_str = &move_str[2..4];
    
    let from = Square::from_algebraic(from_str)?;
    let to = Square::from_algebraic(to_str)?;
    
    // Parse promotion piece if present
    let promo_piece = if move_str.len() > 4 {
        match move_str.chars().nth(4)? {
            'q' => Some(Piece::Queen),
            'r' => Some(Piece::Rook),
            'b' => Some(Piece::Bishop),
            'n' => Some(Piece::Knight),
            _ => None,
        }
    } else {
        None
    };

    // Find the matching legal move
    let moves = board.generate_moves();
    for m in moves.iter() {
        if m.from() == from && m.to() == to {
            // For promotions, also check the promotion piece
            if let Some(p) = promo_piece {
                if m.flag().promotion_piece() == Some(p) {
                    return Some(m);
                }
            } else if m.flag().promotion_piece().is_none() {
                return Some(m);
            }
        }
    }

    None
}

/// Format a move to UCI notation (e.g., "e2e4", "e7e8q")
pub fn format_move(m: Move) -> String {
    m.to_uci()
}
