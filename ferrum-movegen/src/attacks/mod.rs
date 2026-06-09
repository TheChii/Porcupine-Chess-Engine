//! Attack table generation and lookup.

pub mod pawn;
pub mod knight;
pub mod king;
pub mod rays;
pub mod magic;
pub mod between;

pub use pawn::{pawn_attacks, PAWN_ATTACKS};
pub use knight::{knight_attacks, KNIGHT_ATTACKS};
pub use king::{king_attacks, KING_ATTACKS};
pub use magic::{bishop_attacks, rook_attacks};
pub use between::{between, line};
