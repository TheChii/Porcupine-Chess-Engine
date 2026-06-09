//! Piece and Color types.

use core::fmt;
use core::ops::Not;

/// Represents a player color.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    /// All colors.
    pub const ALL: [Color; 2] = [Color::White, Color::Black];

    /// Get the index of this color (0 or 1).
    #[inline(always)]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Check if this is white.
    #[inline(always)]
    pub const fn is_white(self) -> bool {
        matches!(self, Color::White)
    }

    /// Check if this is black.
    #[inline(always)]
    pub const fn is_black(self) -> bool {
        matches!(self, Color::Black)
    }

    /// Get the pawn push direction for this color.
    /// White: +8, Black: -8
    #[inline(always)]
    pub const fn pawn_push(self) -> i8 {
        match self {
            Color::White => 8,
            Color::Black => -8,
        }
    }

    /// Get the back rank for this color.
    #[inline(always)]
    pub const fn back_rank(self) -> super::Rank {
        match self {
            Color::White => super::Rank::R1,
            Color::Black => super::Rank::R8,
        }
    }

    /// Get the pawn starting rank for this color.
    #[inline(always)]
    pub const fn pawn_rank(self) -> super::Rank {
        match self {
            Color::White => super::Rank::R2,
            Color::Black => super::Rank::R7,
        }
    }

    /// Get the promotion rank for this color.
    #[inline(always)]
    pub const fn promotion_rank(self) -> super::Rank {
        match self {
            Color::White => super::Rank::R8,
            Color::Black => super::Rank::R1,
        }
    }
}

impl Not for Color {
    type Output = Color;

    #[inline(always)]
    fn not(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Color::White => write!(f, "w"),
            Color::Black => write!(f, "b"),
        }
    }
}

/// Represents a piece type (without color).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Piece {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl Piece {
    /// All piece types.
    pub const ALL: [Piece; 6] = [
        Piece::Pawn, Piece::Knight, Piece::Bishop,
        Piece::Rook, Piece::Queen, Piece::King,
    ];

    /// Number of piece types.
    pub const COUNT: usize = 6;

    /// Get the index of this piece type (0-5).
    #[inline(always)]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Create from index (0-5).
    #[inline(always)]
    pub const fn from_index(index: u8) -> Option<Piece> {
        match index {
            0 => Some(Piece::Pawn),
            1 => Some(Piece::Knight),
            2 => Some(Piece::Bishop),
            3 => Some(Piece::Rook),
            4 => Some(Piece::Queen),
            5 => Some(Piece::King),
            _ => None,
        }
    }

    /// Create from index, unchecked.
    #[inline(always)]
    pub const unsafe fn from_index_unchecked(index: u8) -> Piece {
        core::mem::transmute(index)
    }

    /// Get piece from FEN character.
    pub const fn from_char(c: char) -> Option<(Piece, Color)> {
        match c {
            'P' => Some((Piece::Pawn, Color::White)),
            'N' => Some((Piece::Knight, Color::White)),
            'B' => Some((Piece::Bishop, Color::White)),
            'R' => Some((Piece::Rook, Color::White)),
            'Q' => Some((Piece::Queen, Color::White)),
            'K' => Some((Piece::King, Color::White)),
            'p' => Some((Piece::Pawn, Color::Black)),
            'n' => Some((Piece::Knight, Color::Black)),
            'b' => Some((Piece::Bishop, Color::Black)),
            'r' => Some((Piece::Rook, Color::Black)),
            'q' => Some((Piece::Queen, Color::Black)),
            'k' => Some((Piece::King, Color::Black)),
            _ => None,
        }
    }

    /// Convert to FEN character.
    #[inline]
    pub const fn to_char(self, color: Color) -> char {
        let base = match self {
            Piece::Pawn => 'p',
            Piece::Knight => 'n',
            Piece::Bishop => 'b',
            Piece::Rook => 'r',
            Piece::Queen => 'q',
            Piece::King => 'k',
        };
        if color.is_white() {
            (base as u8 - 32) as char // uppercase
        } else {
            base
        }
    }

    /// Check if this piece is a slider (bishop, rook, queen).
    #[inline(always)]
    pub const fn is_slider(self) -> bool {
        matches!(self, Piece::Bishop | Piece::Rook | Piece::Queen)
    }

    /// Get promotion piece from index (0-3 = N, B, R, Q).
    #[inline(always)]
    pub const fn from_promotion_index(index: u8) -> Piece {
        match index {
            0 => Piece::Knight,
            1 => Piece::Bishop,
            2 => Piece::Rook,
            _ => Piece::Queen,
        }
    }
}

impl fmt::Display for Piece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = match self {
            Piece::Pawn => 'P',
            Piece::Knight => 'N',
            Piece::Bishop => 'B',
            Piece::Rook => 'R',
            Piece::Queen => 'Q',
            Piece::King => 'K',
        };
        write!(f, "{}", c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_flip() {
        assert_eq!(!Color::White, Color::Black);
        assert_eq!(!Color::Black, Color::White);
    }

    #[test]
    fn test_piece_from_char() {
        assert_eq!(Piece::from_char('K'), Some((Piece::King, Color::White)));
        assert_eq!(Piece::from_char('k'), Some((Piece::King, Color::Black)));
        assert_eq!(Piece::from_char('N'), Some((Piece::Knight, Color::White)));
        assert_eq!(Piece::from_char('x'), None);
    }

    #[test]
    fn test_piece_to_char() {
        assert_eq!(Piece::King.to_char(Color::White), 'K');
        assert_eq!(Piece::King.to_char(Color::Black), 'k');
        assert_eq!(Piece::Queen.to_char(Color::White), 'Q');
    }
}
