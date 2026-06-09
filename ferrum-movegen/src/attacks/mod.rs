//! Attack table generation and lookup.

pub mod between;
pub mod king;
pub mod knight;
pub mod magic;
pub mod pawn;
pub mod rays;

pub use between::{between, line};
pub use king::{king_attacks, KING_ATTACKS};
pub use knight::{knight_attacks, KNIGHT_ATTACKS};
pub use magic::{bishop_attacks, rook_attacks};
pub use pawn::{pawn_attacks, PAWN_ATTACKS};
