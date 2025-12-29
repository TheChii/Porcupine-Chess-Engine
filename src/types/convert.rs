//! Conversion traits between `movegen` crate and `nnue` crate types.
//!
//! The `movegen` crate and `nnue` crate have their own Square, Piece, and Color types.
//! This module provides zero-cost conversions between them.

use movegen::{Square as MovegenSquare, Piece as MovegenPiece, Color as MovegenColor};
use nnue::{Square as NnueSquare, Piece as NnuePiece, Color as NnueColor};

/// Trait for converting movegen crate types to nnue crate types.
///
/// Implementations are `#[inline]` for zero-cost abstraction.
pub trait ToNnue {
    type Output;
    fn to_nnue(self) -> Self::Output;
}

impl ToNnue for MovegenSquare {
    type Output = NnueSquare;

    #[inline]
    fn to_nnue(self) -> NnueSquare {
        // Both crates use A1=0, H8=63 ordering
        NnueSquare::from_index(self.index() as usize)
    }
}

impl ToNnue for MovegenPiece {
    type Output = NnuePiece;

    #[inline]
    fn to_nnue(self) -> NnuePiece {
        // Piece ordering: Pawn, Knight, Bishop, Rook, Queen, King
        match self {
            MovegenPiece::Pawn => NnuePiece::Pawn,
            MovegenPiece::Knight => NnuePiece::Knight,
            MovegenPiece::Bishop => NnuePiece::Bishop,
            MovegenPiece::Rook => NnuePiece::Rook,
            MovegenPiece::Queen => NnuePiece::Queen,
            MovegenPiece::King => NnuePiece::King,
        }
    }
}

impl ToNnue for MovegenColor {
    type Output = NnueColor;

    #[inline]
    fn to_nnue(self) -> NnueColor {
        match self {
            MovegenColor::White => NnueColor::White,
            MovegenColor::Black => NnueColor::Black,
        }
    }
}

/// Helper to get the opposite color in nnue terms
#[inline]
#[allow(dead_code)]
pub fn nnue_color_flip(c: NnueColor) -> NnueColor {
    match c {
        NnueColor::White => NnueColor::Black,
        NnueColor::Black => NnueColor::White,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_conversion() {
        // Test a few key squares
        assert_eq!(MovegenSquare::A1.to_nnue(), NnueSquare::A1);
        assert_eq!(MovegenSquare::E4.to_nnue(), NnueSquare::E4);
        assert_eq!(MovegenSquare::H8.to_nnue(), NnueSquare::H8);
    }

    #[test]
    fn test_piece_conversion() {
        assert_eq!(MovegenPiece::Pawn.to_nnue(), NnuePiece::Pawn);
        assert_eq!(MovegenPiece::King.to_nnue(), NnuePiece::King);
    }

    #[test]
    fn test_color_conversion() {
        assert_eq!(MovegenColor::White.to_nnue(), NnueColor::White);
        assert_eq!(MovegenColor::Black.to_nnue(), NnueColor::Black);
    }
}
