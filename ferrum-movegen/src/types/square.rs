//! Square, File, and Rank types for chess board positions.
//!
//! Uses Little-Endian Rank-File Mapping:
//! - A1 = 0, B1 = 1, ..., H1 = 7
//! - A2 = 8, ..., H8 = 63

use core::fmt;

/// Represents a file (column) on the chess board (A-H).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum File {
    A = 0, B = 1, C = 2, D = 3,
    E = 4, F = 5, G = 6, H = 7,
}

impl File {
    /// All files in order A-H.
    pub const ALL: [File; 8] = [
        File::A, File::B, File::C, File::D,
        File::E, File::F, File::G, File::H,
    ];

    /// Create a File from an index (0-7).
    #[inline(always)]
    pub const fn from_index(index: u8) -> Option<File> {
        match index {
            0 => Some(File::A), 1 => Some(File::B),
            2 => Some(File::C), 3 => Some(File::D),
            4 => Some(File::E), 5 => Some(File::F),
            6 => Some(File::G), 7 => Some(File::H),
            _ => None,
        }
    }

    /// Create a File from an index, unchecked.
    #[inline(always)]
    pub const unsafe fn from_index_unchecked(index: u8) -> File {
        core::mem::transmute(index)
    }

    /// Get the index of this file (0-7).
    #[inline(always)]
    pub const fn index(self) -> u8 {
        self as u8
    }

    /// Get file from character ('a'-'h' or 'A'-'H').
    #[inline]
    pub const fn from_char(c: char) -> Option<File> {
        match c {
            'a' | 'A' => Some(File::A),
            'b' | 'B' => Some(File::B),
            'c' | 'C' => Some(File::C),
            'd' | 'D' => Some(File::D),
            'e' | 'E' => Some(File::E),
            'f' | 'F' => Some(File::F),
            'g' | 'G' => Some(File::G),
            'h' | 'H' => Some(File::H),
            _ => None,
        }
    }

    /// Convert to lowercase character.
    #[inline(always)]
    pub const fn to_char(self) -> char {
        (b'a' + self as u8) as char
    }
}

impl fmt::Display for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

/// Represents a rank (row) on the chess board (1-8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Rank {
    R1 = 0, R2 = 1, R3 = 2, R4 = 3,
    R5 = 4, R6 = 5, R7 = 6, R8 = 7,
}

impl Rank {
    /// All ranks in order 1-8.
    pub const ALL: [Rank; 8] = [
        Rank::R1, Rank::R2, Rank::R3, Rank::R4,
        Rank::R5, Rank::R6, Rank::R7, Rank::R8,
    ];

    /// Create a Rank from an index (0-7).
    #[inline(always)]
    pub const fn from_index(index: u8) -> Option<Rank> {
        match index {
            0 => Some(Rank::R1), 1 => Some(Rank::R2),
            2 => Some(Rank::R3), 3 => Some(Rank::R4),
            4 => Some(Rank::R5), 5 => Some(Rank::R6),
            6 => Some(Rank::R7), 7 => Some(Rank::R8),
            _ => None,
        }
    }

    /// Create a Rank from an index, unchecked.
    #[inline(always)]
    pub const unsafe fn from_index_unchecked(index: u8) -> Rank {
        core::mem::transmute(index)
    }

    /// Get the index of this rank (0-7).
    #[inline(always)]
    pub const fn index(self) -> u8 {
        self as u8
    }

    /// Get rank from character ('1'-'8').
    #[inline]
    pub const fn from_char(c: char) -> Option<Rank> {
        match c {
            '1' => Some(Rank::R1), '2' => Some(Rank::R2),
            '3' => Some(Rank::R3), '4' => Some(Rank::R4),
            '5' => Some(Rank::R5), '6' => Some(Rank::R6),
            '7' => Some(Rank::R7), '8' => Some(Rank::R8),
            _ => None,
        }
    }

    /// Convert to character.
    #[inline(always)]
    pub const fn to_char(self) -> char {
        (b'1' + self as u8) as char
    }

    /// Flip the rank (for black's perspective).
    #[inline(always)]
    pub const fn flip(self) -> Rank {
        unsafe { Rank::from_index_unchecked(7 - self as u8) }
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

/// Represents a square on the chess board (A1-H8).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Square(u8);

impl Square {
    // Named constants for all 64 squares
    pub const A1: Square = Square(0);  pub const B1: Square = Square(1);
    pub const C1: Square = Square(2);  pub const D1: Square = Square(3);
    pub const E1: Square = Square(4);  pub const F1: Square = Square(5);
    pub const G1: Square = Square(6);  pub const H1: Square = Square(7);
    pub const A2: Square = Square(8);  pub const B2: Square = Square(9);
    pub const C2: Square = Square(10); pub const D2: Square = Square(11);
    pub const E2: Square = Square(12); pub const F2: Square = Square(13);
    pub const G2: Square = Square(14); pub const H2: Square = Square(15);
    pub const A3: Square = Square(16); pub const B3: Square = Square(17);
    pub const C3: Square = Square(18); pub const D3: Square = Square(19);
    pub const E3: Square = Square(20); pub const F3: Square = Square(21);
    pub const G3: Square = Square(22); pub const H3: Square = Square(23);
    pub const A4: Square = Square(24); pub const B4: Square = Square(25);
    pub const C4: Square = Square(26); pub const D4: Square = Square(27);
    pub const E4: Square = Square(28); pub const F4: Square = Square(29);
    pub const G4: Square = Square(30); pub const H4: Square = Square(31);
    pub const A5: Square = Square(32); pub const B5: Square = Square(33);
    pub const C5: Square = Square(34); pub const D5: Square = Square(35);
    pub const E5: Square = Square(36); pub const F5: Square = Square(37);
    pub const G5: Square = Square(38); pub const H5: Square = Square(39);
    pub const A6: Square = Square(40); pub const B6: Square = Square(41);
    pub const C6: Square = Square(42); pub const D6: Square = Square(43);
    pub const E6: Square = Square(44); pub const F6: Square = Square(45);
    pub const G6: Square = Square(46); pub const H6: Square = Square(47);
    pub const A7: Square = Square(48); pub const B7: Square = Square(49);
    pub const C7: Square = Square(50); pub const D7: Square = Square(51);
    pub const E7: Square = Square(52); pub const F7: Square = Square(53);
    pub const G7: Square = Square(54); pub const H7: Square = Square(55);
    pub const A8: Square = Square(56); pub const B8: Square = Square(57);
    pub const C8: Square = Square(58); pub const D8: Square = Square(59);
    pub const E8: Square = Square(60); pub const F8: Square = Square(61);
    pub const G8: Square = Square(62); pub const H8: Square = Square(63);

    /// Number of squares on the board.
    pub const COUNT: usize = 64;

    /// Create a square from file and rank.
    #[inline(always)]
    pub const fn from_file_rank(file: File, rank: Rank) -> Square {
        Square((rank as u8) * 8 + (file as u8))
    }

    /// Create a square from an index (0-63).
    #[inline(always)]
    pub const fn from_index(index: u8) -> Option<Square> {
        if index < 64 {
            Some(Square(index))
        } else {
            None
        }
    }

    /// Create a square from an index, unchecked.
    #[inline(always)]
    pub const unsafe fn from_index_unchecked(index: u8) -> Square {
        Square(index)
    }

    /// Get the index of this square (0-63).
    #[inline(always)]
    pub const fn index(self) -> u8 {
        self.0
    }

    /// Get the file of this square.
    #[inline(always)]
    pub const fn file(self) -> File {
        unsafe { File::from_index_unchecked(self.0 & 7) }
    }

    /// Get the rank of this square.
    #[inline(always)]
    pub const fn rank(self) -> Rank {
        unsafe { Rank::from_index_unchecked(self.0 >> 3) }
    }

    /// Convert to a bitboard with only this square set.
    #[inline(always)]
    pub const fn to_bb(self) -> u64 {
        1u64 << self.0
    }

    /// Flip the square vertically (for black's perspective).
    #[inline(always)]
    pub const fn flip_vertical(self) -> Square {
        Square(self.0 ^ 56)
    }

    /// Flip the square horizontally.
    #[inline(always)]
    pub const fn flip_horizontal(self) -> Square {
        Square(self.0 ^ 7)
    }

    /// Get the square to the north (if valid).
    #[inline(always)]
    pub const fn north(self) -> Option<Square> {
        if self.0 < 56 {
            Some(Square(self.0 + 8))
        } else {
            None
        }
    }

    /// Get the square to the south (if valid).
    #[inline(always)]
    pub const fn south(self) -> Option<Square> {
        if self.0 >= 8 {
            Some(Square(self.0 - 8))
        } else {
            None
        }
    }

    /// Get the square to the east (if valid).
    #[inline(always)]
    pub const fn east(self) -> Option<Square> {
        if (self.0 & 7) < 7 {
            Some(Square(self.0 + 1))
        } else {
            None
        }
    }

    /// Get the square to the west (if valid).
    #[inline(always)]
    pub const fn west(self) -> Option<Square> {
        if (self.0 & 7) > 0 {
            Some(Square(self.0 - 1))
        } else {
            None
        }
    }

    /// Parse a square from algebraic notation (e.g., "e4").
    pub fn from_algebraic(s: &str) -> Option<Square> {
        let bytes = s.as_bytes();
        if bytes.len() != 2 {
            return None;
        }
        let file = File::from_char(bytes[0] as char)?;
        let rank = Rank::from_char(bytes[1] as char)?;
        Some(Square::from_file_rank(file, rank))
    }

    /// Convert to algebraic notation.
    #[inline]
    pub fn to_algebraic(self) -> [char; 2] {
        [self.file().to_char(), self.rank().to_char()]
    }
}

impl fmt::Debug for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [file, rank] = self.to_algebraic();
        write!(f, "{}{}", file, rank)
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [file, rank] = self.to_algebraic();
        write!(f, "{}{}", file, rank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_from_file_rank() {
        assert_eq!(Square::from_file_rank(File::A, Rank::R1), Square::A1);
        assert_eq!(Square::from_file_rank(File::E, Rank::R4), Square::E4);
        assert_eq!(Square::from_file_rank(File::H, Rank::R8), Square::H8);
    }

    #[test]
    fn test_square_file_rank() {
        assert_eq!(Square::E4.file(), File::E);
        assert_eq!(Square::E4.rank(), Rank::R4);
        assert_eq!(Square::A1.file(), File::A);
        assert_eq!(Square::A1.rank(), Rank::R1);
    }

    #[test]
    fn test_square_algebraic() {
        assert_eq!(Square::from_algebraic("e4"), Some(Square::E4));
        assert_eq!(Square::from_algebraic("a1"), Some(Square::A1));
        assert_eq!(Square::from_algebraic("h8"), Some(Square::H8));
        assert_eq!(Square::from_algebraic("i9"), None);
    }

    #[test]
    fn test_square_flip() {
        assert_eq!(Square::A1.flip_vertical(), Square::A8);
        assert_eq!(Square::E4.flip_vertical(), Square::E5);
        assert_eq!(Square::H8.flip_vertical(), Square::H1);
    }

    #[test]
    fn test_square_directions() {
        assert_eq!(Square::E4.north(), Some(Square::E5));
        assert_eq!(Square::E4.south(), Some(Square::E3));
        assert_eq!(Square::E4.east(), Some(Square::F4));
        assert_eq!(Square::E4.west(), Some(Square::D4));
        assert_eq!(Square::A8.north(), None);
        assert_eq!(Square::A1.south(), None);
        assert_eq!(Square::H4.east(), None);
        assert_eq!(Square::A4.west(), None);
    }
}
