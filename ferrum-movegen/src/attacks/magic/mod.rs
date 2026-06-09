//! Magic bitboard infrastructure for sliding piece attacks.

#![allow(long_running_const_eval)]

mod rook;
mod bishop;

#[cfg(feature = "pext")]
mod pext;

// When PEXT is enabled, use PEXT-based attacks
#[cfg(feature = "pext")]
pub use pext::{rook_attacks_pext as rook_attacks, bishop_attacks_pext as bishop_attacks};

// Otherwise, use magic bitboard attacks
#[cfg(not(feature = "pext"))]
pub use rook::rook_attacks;
#[cfg(not(feature = "pext"))]
pub use bishop::bishop_attacks;

pub use rook::ROOK_MAGICS;
pub use bishop::BISHOP_MAGICS;

use crate::bitboard::Bitboard;
use crate::types::Square;

/// Magic entry for a single square.
#[derive(Clone, Copy)]
pub struct Magic {
    /// Relevant occupancy mask (excludes edge squares).
    pub mask: Bitboard,
    /// Magic number for hash computation.
    pub magic: u64,
    /// Pointer/offset into attack table.
    pub offset: usize,
    /// Shift amount (64 - bits).
    pub shift: u8,
}

impl Magic {
    /// Empty magic entry.
    pub const EMPTY: Magic = Magic {
        mask: Bitboard::EMPTY,
        magic: 0,
        offset: 0,
        shift: 0,
    };

    /// Compute index into attack table.
    #[inline(always)]
    pub fn index(&self, occ: Bitboard) -> usize {
        let relevant = occ.0 & self.mask.0;
        let hash = relevant.wrapping_mul(self.magic);
        self.offset + ((hash >> self.shift) as usize)
    }
}

/// Compute relevant occupancy mask for a rook.
pub const fn rook_mask(sq: Square) -> Bitboard {
    let file = sq.index() & 7;
    let rank = sq.index() >> 3;
    let mut mask = 0u64;
    
    // North (exclude rank 8)
    let mut r = rank + 1;
    while r < 7 {
        mask |= 1u64 << (r * 8 + file);
        r += 1;
    }
    
    // South (exclude rank 1)
    let mut r = rank;
    while r > 1 {
        r -= 1;
        mask |= 1u64 << (r * 8 + file);
    }
    
    // East (exclude file H)
    let mut f = file + 1;
    while f < 7 {
        mask |= 1u64 << (rank * 8 + f);
        f += 1;
    }
    
    // West (exclude file A)
    let mut f = file;
    while f > 1 {
        f -= 1;
        mask |= 1u64 << (rank * 8 + f);
    }
    
    Bitboard(mask)
}

/// Compute relevant occupancy mask for a bishop.
pub const fn bishop_mask(sq: Square) -> Bitboard {
    let file = sq.index() & 7;
    let rank = sq.index() >> 3;
    let mut mask = 0u64;
    
    // Northeast (exclude edges)
    let mut r = rank + 1;
    let mut f = file + 1;
    while r < 7 && f < 7 {
        mask |= 1u64 << (r * 8 + f);
        r += 1;
        f += 1;
    }
    
    // Northwest
    let mut r = rank + 1;
    let mut f = file;
    while r < 7 && f > 1 {
        f -= 1;
        mask |= 1u64 << (r * 8 + f);
        r += 1;
    }
    
    // Southeast
    let mut r = rank;
    let mut f = file + 1;
    while r > 1 && f < 7 {
        r -= 1;
        mask |= 1u64 << (r * 8 + f);
        f += 1;
    }
    
    // Southwest
    let mut r = rank;
    let mut f = file;
    while r > 1 && f > 1 {
        r -= 1;
        f -= 1;
        mask |= 1u64 << (r * 8 + f);
    }
    
    Bitboard(mask)
}

/// Compute rook attacks given occupancy (slow reference implementation).
pub const fn rook_attacks_slow(sq: Square, occ: Bitboard) -> Bitboard {
    let file = sq.index() & 7;
    let rank = sq.index() >> 3;
    let mut attacks = 0u64;
    
    // North
    let mut r = rank + 1;
    while r < 8 {
        let sq_bit = 1u64 << (r * 8 + file);
        attacks |= sq_bit;
        if (occ.0 & sq_bit) != 0 { break; }
        r += 1;
    }
    
    // South
    let mut r = rank;
    while r > 0 {
        r -= 1;
        let sq_bit = 1u64 << (r * 8 + file);
        attacks |= sq_bit;
        if (occ.0 & sq_bit) != 0 { break; }
    }
    
    // East
    let mut f = file + 1;
    while f < 8 {
        let sq_bit = 1u64 << (rank * 8 + f);
        attacks |= sq_bit;
        if (occ.0 & sq_bit) != 0 { break; }
        f += 1;
    }
    
    // West
    let mut f = file;
    while f > 0 {
        f -= 1;
        let sq_bit = 1u64 << (rank * 8 + f);
        attacks |= sq_bit;
        if (occ.0 & sq_bit) != 0 { break; }
    }
    
    Bitboard(attacks)
}

/// Compute bishop attacks given occupancy (slow reference implementation).
pub const fn bishop_attacks_slow(sq: Square, occ: Bitboard) -> Bitboard {
    let file = sq.index() & 7;
    let rank = sq.index() >> 3;
    let mut attacks = 0u64;
    
    // Northeast
    let mut r = rank + 1;
    let mut f = file + 1;
    while r < 8 && f < 8 {
        let sq_bit = 1u64 << (r * 8 + f);
        attacks |= sq_bit;
        if (occ.0 & sq_bit) != 0 { break; }
        r += 1;
        f += 1;
    }
    
    // Northwest
    let mut r = rank + 1;
    let mut f = file;
    while r < 8 && f > 0 {
        f -= 1;
        let sq_bit = 1u64 << (r * 8 + f);
        attacks |= sq_bit;
        if (occ.0 & sq_bit) != 0 { break; }
        r += 1;
    }
    
    // Southeast
    let mut r = rank;
    let mut f = file + 1;
    while r > 0 && f < 8 {
        r -= 1;
        let sq_bit = 1u64 << (r * 8 + f);
        attacks |= sq_bit;
        if (occ.0 & sq_bit) != 0 { break; }
        f += 1;
    }
    
    // Southwest
    let mut r = rank;
    let mut f = file;
    while r > 0 && f > 0 {
        r -= 1;
        f -= 1;
        let sq_bit = 1u64 << (r * 8 + f);
        attacks |= sq_bit;
        if (occ.0 & sq_bit) != 0 { break; }
    }
    
    Bitboard(attacks)
}

/// Enumerate all occupancy configurations for a mask.
#[inline]
pub fn enumerate_occupancies(mask: Bitboard) -> impl Iterator<Item = Bitboard> {
    let mask_val = mask.0;
    let count = 1u64 << mask.count();
    (0..count).map(move |i| {
        // Map index bits to mask bits using carry-rippler technique
        let mut occ = 0u64;
        let mut m = mask_val;
        let mut idx = i;
        while m != 0 {
            let lsb = m & m.wrapping_neg();
            if (idx & 1) != 0 {
                occ |= lsb;
            }
            m &= m - 1;
            idx >>= 1;
        }
        Bitboard(occ)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rook_mask() {
        // Rook on e4: mask should exclude edges
        let mask = rook_mask(Square::E4);
        assert!(!mask.contains(Square::E1)); // Edge
        assert!(!mask.contains(Square::E8)); // Edge
        assert!(!mask.contains(Square::A4)); // Edge
        assert!(!mask.contains(Square::H4)); // Edge
        assert!(mask.contains(Square::E5));
        assert!(mask.contains(Square::F4));
    }

    #[test]
    fn test_bishop_mask() {
        let mask = bishop_mask(Square::E4);
        assert!(!mask.contains(Square::H7)); // Edge
        assert!(!mask.contains(Square::A8)); // Edge
        assert!(mask.contains(Square::F5));
        assert!(mask.contains(Square::D3));
    }

    #[test]
    fn test_rook_attacks_slow() {
        let occ = Bitboard::from_square(Square::E6);
        let attacks = rook_attacks_slow(Square::E4, occ);
        assert!(attacks.contains(Square::E5));
        assert!(attacks.contains(Square::E6)); // Blocker included
        assert!(!attacks.contains(Square::E7)); // Blocked
    }
}
