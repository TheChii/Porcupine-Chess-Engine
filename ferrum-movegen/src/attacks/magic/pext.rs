//! PEXT-based sliding piece attacks (BMI2).
//!
//! PEXT (Parallel Bit Extract) provides O(1) perfect hashing for sliding attacks.
//! This is faster than magic bitboards on CPUs with native BMI2 support.

use crate::bitboard::Bitboard;
use crate::types::Square;
use super::{rook_mask, bishop_mask, rook_attacks_slow, bishop_attacks_slow};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_pext_u64;

/// PEXT entry for a single square.
#[derive(Clone, Copy)]
pub struct PextEntry {
    /// Relevant occupancy mask.
    pub mask: Bitboard,
    /// Offset into attack table.
    pub offset: usize,
}

impl PextEntry {
    pub const EMPTY: PextEntry = PextEntry {
        mask: Bitboard::EMPTY,
        offset: 0,
    };
}

/// Rook PEXT entries.
pub static ROOK_PEXT: [PextEntry; 64] = init_rook_pext_entries();

/// Bishop PEXT entries.
pub static BISHOP_PEXT: [PextEntry; 64] = init_bishop_pext_entries();

/// Rook attack table for PEXT.
/// Size: sum of 2^popcount(mask) for each square ≈ 102KB
static ROOK_PEXT_ATTACKS: [Bitboard; 102400] = init_rook_pext_attacks();

/// Bishop attack table for PEXT.
/// Size: sum of 2^popcount(mask) for each square ≈ 5KB
static BISHOP_PEXT_ATTACKS: [Bitboard; 5248] = init_bishop_pext_attacks();

/// Get rook attacks using PEXT.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn rook_attacks_pext(sq: Square, occ: Bitboard) -> Bitboard {
    let entry = &ROOK_PEXT[sq.index() as usize];
    let index = unsafe { _pext_u64(occ.0, entry.mask.0) } as usize;
    ROOK_PEXT_ATTACKS[entry.offset + index]
}

/// Get bishop attacks using PEXT.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
pub fn bishop_attacks_pext(sq: Square, occ: Bitboard) -> Bitboard {
    let entry = &BISHOP_PEXT[sq.index() as usize];
    let index = unsafe { _pext_u64(occ.0, entry.mask.0) } as usize;
    BISHOP_PEXT_ATTACKS[entry.offset + index]
}

/// Fallback for non-x86_64 targets.
#[cfg(not(target_arch = "x86_64"))]
pub fn rook_attacks_pext(sq: Square, occ: Bitboard) -> Bitboard {
    super::rook::rook_attacks(sq, occ)
}

#[cfg(not(target_arch = "x86_64"))]
pub fn bishop_attacks_pext(sq: Square, occ: Bitboard) -> Bitboard {
    super::bishop::bishop_attacks(sq, occ)
}

const fn init_rook_pext_entries() -> [PextEntry; 64] {
    let mut entries = [PextEntry::EMPTY; 64];
    let mut offset = 0usize;
    let mut sq = 0u8;
    
    while sq < 64 {
        let square = unsafe { Square::from_index_unchecked(sq) };
        let mask = rook_mask(square);
        let size = 1usize << mask.count();
        
        entries[sq as usize] = PextEntry { mask, offset };
        offset += size;
        sq += 1;
    }
    
    entries
}

const fn init_bishop_pext_entries() -> [PextEntry; 64] {
    let mut entries = [PextEntry::EMPTY; 64];
    let mut offset = 0usize;
    let mut sq = 0u8;
    
    while sq < 64 {
        let square = unsafe { Square::from_index_unchecked(sq) };
        let mask = bishop_mask(square);
        let size = 1usize << mask.count();
        
        entries[sq as usize] = PextEntry { mask, offset };
        offset += size;
        sq += 1;
    }
    
    entries
}

const fn init_rook_pext_attacks() -> [Bitboard; 102400] {
    let mut attacks = [Bitboard::EMPTY; 102400];
    let mut offset = 0usize;
    let mut sq = 0u8;
    
    while sq < 64 {
        let square = unsafe { Square::from_index_unchecked(sq) };
        let mask = rook_mask(square);
        let bits = mask.count();
        let size = 1usize << bits;
        
        let mut idx = 0usize;
        while idx < size {
            // Reconstruct occupancy from PEXT index
            let occ = pdep_const(idx as u64, mask.0);
            let attack = rook_attacks_slow(square, Bitboard(occ));
            attacks[offset + idx] = attack;
            idx += 1;
        }
        
        offset += size;
        sq += 1;
    }
    
    attacks
}

const fn init_bishop_pext_attacks() -> [Bitboard; 5248] {
    let mut attacks = [Bitboard::EMPTY; 5248];
    let mut offset = 0usize;
    let mut sq = 0u8;
    
    while sq < 64 {
        let square = unsafe { Square::from_index_unchecked(sq) };
        let mask = bishop_mask(square);
        let bits = mask.count();
        let size = 1usize << bits;
        
        let mut idx = 0usize;
        while idx < size {
            let occ = pdep_const(idx as u64, mask.0);
            let attack = bishop_attacks_slow(square, Bitboard(occ));
            attacks[offset + idx] = attack;
            idx += 1;
        }
        
        offset += size;
        sq += 1;
    }
    
    attacks
}

/// Const-compatible PDEP implementation.
const fn pdep_const(src: u64, mask: u64) -> u64 {
    let mut result = 0u64;
    let mut m = mask;
    let mut s = src;
    
    while m != 0 {
        let lsb = m & m.wrapping_neg();
        if (s & 1) != 0 {
            result |= lsb;
        }
        m &= m - 1;
        s >>= 1;
    }
    
    result
}
