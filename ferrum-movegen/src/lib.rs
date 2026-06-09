//! # Chess Move Generator
//!
//! A high-performance, modular legal chess move generator using bitboards
//! and magic bitboards for sliding piece attacks.
//!
//! ## Features
//! - `std` (default): Enable standard library features
//! - `pext`: Enable BMI2 PEXT instructions for faster sliding attacks
//!
//! ## Example
//! ```
//! use movegen::Board;
//!
//! let board = Board::default();
//! let moves = board.generate_moves();
//! println!("Legal moves: {}", moves.len());
//! ```
//!
//! ## Performance
//! - ~468M nodes/second in perft benchmarks
//! - Compile-time generated attack tables
//! - Stack-allocated move lists (no heap allocations in hot paths)

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::missing_safety_doc)]  // Safety docs for internal unsafe fns
#![allow(clippy::if_same_then_else)]   // Sometimes clearer to have explicit branches

pub mod types;
pub mod bitboard;
pub mod attacks;
pub mod board;
pub mod movegen;

#[cfg(feature = "std")]
pub mod testing;

// Re-export commonly used types
pub use types::{Square, File, Rank, Color, Piece, CastleRights};
pub use bitboard::Bitboard;
pub use board::Board;
pub use movegen::{Move, MoveList, MoveFlag};
