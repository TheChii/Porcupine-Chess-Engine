//! Castling rights representation.

use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not};

/// Castling rights as a 4-bit mask.
///
/// Bit layout:
/// - Bit 0: White kingside (K)
/// - Bit 1: White queenside (Q)
/// - Bit 2: Black kingside (k)
/// - Bit 3: Black queenside (q)
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct CastleRights(u8);

impl CastleRights {
    /// No castling rights.
    pub const NONE: CastleRights = CastleRights(0);

    /// All castling rights.
    pub const ALL: CastleRights = CastleRights(0b1111);

    /// White kingside.
    pub const WHITE_KINGSIDE: CastleRights = CastleRights(0b0001);

    /// White queenside.
    pub const WHITE_QUEENSIDE: CastleRights = CastleRights(0b0010);

    /// Black kingside.
    pub const BLACK_KINGSIDE: CastleRights = CastleRights(0b0100);

    /// Black queenside.
    pub const BLACK_QUEENSIDE: CastleRights = CastleRights(0b1000);

    /// All white castling rights.
    pub const WHITE: CastleRights = CastleRights(0b0011);

    /// All black castling rights.
    pub const BLACK: CastleRights = CastleRights(0b1100);

    /// Create from raw bits.
    #[inline(always)]
    pub const fn from_bits(bits: u8) -> CastleRights {
        CastleRights(bits & 0b1111)
    }

    /// Get raw bits.
    #[inline(always)]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Check if no castling rights remain.
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Check if any castling rights remain.
    #[inline(always)]
    pub const fn any(self) -> bool {
        self.0 != 0
    }

    /// Check if white can castle kingside.
    #[inline(always)]
    pub const fn has_white_kingside(self) -> bool {
        (self.0 & 0b0001) != 0
    }

    /// Check if white can castle queenside.
    #[inline(always)]
    pub const fn has_white_queenside(self) -> bool {
        (self.0 & 0b0010) != 0
    }

    /// Check if black can castle kingside.
    #[inline(always)]
    pub const fn has_black_kingside(self) -> bool {
        (self.0 & 0b0100) != 0
    }

    /// Check if black can castle queenside.
    #[inline(always)]
    pub const fn has_black_queenside(self) -> bool {
        (self.0 & 0b1000) != 0
    }

    /// Check if the given color can castle kingside.
    #[inline(always)]
    pub const fn has_kingside(self, color: super::Color) -> bool {
        match color {
            super::Color::White => self.has_white_kingside(),
            super::Color::Black => self.has_black_kingside(),
        }
    }

    /// Check if the given color can castle queenside.
    #[inline(always)]
    pub const fn has_queenside(self, color: super::Color) -> bool {
        match color {
            super::Color::White => self.has_white_queenside(),
            super::Color::Black => self.has_black_queenside(),
        }
    }

    /// Remove a specific right.
    #[inline(always)]
    pub const fn remove(self, right: CastleRights) -> CastleRights {
        CastleRights(self.0 & !right.0)
    }

    /// Add a specific right.
    #[inline(always)]
    pub const fn add(self, right: CastleRights) -> CastleRights {
        CastleRights(self.0 | right.0)
    }

    /// Parse from FEN castling string (e.g., "KQkq", "Kq", "-").
    pub fn from_fen(s: &str) -> Option<CastleRights> {
        if s == "-" {
            return Some(CastleRights::NONE);
        }

        let mut rights = CastleRights::NONE;
        for c in s.chars() {
            match c {
                'K' => rights = rights.add(CastleRights::WHITE_KINGSIDE),
                'Q' => rights = rights.add(CastleRights::WHITE_QUEENSIDE),
                'k' => rights = rights.add(CastleRights::BLACK_KINGSIDE),
                'q' => rights = rights.add(CastleRights::BLACK_QUEENSIDE),
                _ => return None,
            }
        }
        Some(rights)
    }

    /// Convert to FEN string.
    pub fn to_fen(self) -> &'static str {
        const STRINGS: [&str; 16] = [
            "-", "K", "Q", "KQ", "k", "Kk", "Qk", "KQk",
            "q", "Kq", "Qq", "KQq", "kq", "Kkq", "Qkq", "KQkq",
        ];
        STRINGS[self.0 as usize]
    }

    /// Mask to apply when a rook or king moves from/to a square.
    /// Returns the bits to clear from castling rights.
    #[inline]
    pub const fn update_mask(sq: super::Square) -> CastleRights {
        // Precomputed masks for squares that affect castling
        const MASKS: [u8; 64] = {
            let mut masks = [0u8; 64];
            masks[0] = 0b0010;   // A1 - white queenside rook
            masks[4] = 0b0011;   // E1 - white king
            masks[7] = 0b0001;   // H1 - white kingside rook
            masks[56] = 0b1000;  // A8 - black queenside rook
            masks[60] = 0b1100;  // E8 - black king
            masks[63] = 0b0100;  // H8 - black kingside rook
            masks
        };
        CastleRights(MASKS[sq.index() as usize])
    }
}

impl BitAnd for CastleRights {
    type Output = CastleRights;
    #[inline(always)]
    fn bitand(self, rhs: CastleRights) -> CastleRights {
        CastleRights(self.0 & rhs.0)
    }
}

impl BitAndAssign for CastleRights {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: CastleRights) {
        self.0 &= rhs.0;
    }
}

impl BitOr for CastleRights {
    type Output = CastleRights;
    #[inline(always)]
    fn bitor(self, rhs: CastleRights) -> CastleRights {
        CastleRights(self.0 | rhs.0)
    }
}

impl BitOrAssign for CastleRights {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: CastleRights) {
        self.0 |= rhs.0;
    }
}

impl Not for CastleRights {
    type Output = CastleRights;
    #[inline(always)]
    fn not(self) -> CastleRights {
        CastleRights(!self.0 & 0b1111)
    }
}

impl fmt::Debug for CastleRights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CastleRights({})", self.to_fen())
    }
}

impl fmt::Display for CastleRights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_fen())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_castling_from_fen() {
        assert_eq!(CastleRights::from_fen("-"), Some(CastleRights::NONE));
        assert_eq!(CastleRights::from_fen("KQkq"), Some(CastleRights::ALL));
        assert_eq!(CastleRights::from_fen("K"), Some(CastleRights::WHITE_KINGSIDE));
        assert_eq!(CastleRights::from_fen("Kq"), 
            Some(CastleRights::WHITE_KINGSIDE | CastleRights::BLACK_QUEENSIDE));
    }

    #[test]
    fn test_castling_to_fen() {
        assert_eq!(CastleRights::NONE.to_fen(), "-");
        assert_eq!(CastleRights::ALL.to_fen(), "KQkq");
        assert_eq!(CastleRights::WHITE.to_fen(), "KQ");
    }

    #[test]
    fn test_castling_has() {
        assert!(CastleRights::ALL.has_white_kingside());
        assert!(CastleRights::ALL.has_black_queenside());
        assert!(!CastleRights::WHITE.has_black_kingside());
    }

    #[test]
    fn test_castling_remove() {
        let rights = CastleRights::ALL.remove(CastleRights::WHITE_KINGSIDE);
        assert!(!rights.has_white_kingside());
        assert!(rights.has_white_queenside());
    }
}
