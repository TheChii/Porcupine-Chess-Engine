//! Rook magic bitboards.

use super::{Magic, rook_mask, rook_attacks_slow};
use crate::bitboard::Bitboard;
use crate::types::Square;

/// Rook magic numbers (precomputed).
/// These magic numbers have been found through trial and error.
const ROOK_MAGIC_NUMBERS: [u64; 64] = [
    0x0080001020400080, 0x0040001000200040, 0x0080081000200080, 0x0080040800100080,
    0x0080020400080080, 0x0080010200040080, 0x0080008001000200, 0x0080002040800100,
    0x0000800020400080, 0x0000400020005000, 0x0000801000200080, 0x0000800800100080,
    0x0000800400080080, 0x0000800200040080, 0x0000800100020080, 0x0000800040800100,
    0x0000208000400080, 0x0000404000201000, 0x0000808010002000, 0x0000808008001000,
    0x0000808004000800, 0x0000808002000400, 0x0000010100020004, 0x0000020000408104,
    0x0000208080004000, 0x0000200040005000, 0x0000100080200080, 0x0000080080100080,
    0x0000040080080080, 0x0000020080040080, 0x0000010080800200, 0x0000800080004100,
    0x0000204000800080, 0x0000200040401000, 0x0000100080802000, 0x0000080080801000,
    0x0000040080800800, 0x0000020080800400, 0x0000020001010004, 0x0000800040800100,
    0x0000204000808000, 0x0000200040008080, 0x0000100020008080, 0x0000080010008080,
    0x0000040008008080, 0x0000020004008080, 0x0000010002008080, 0x0000004081020004,
    0x0000204000800080, 0x0000200040008080, 0x0000100020008080, 0x0000080010008080,
    0x0000040008008080, 0x0000020004008080, 0x0000800100020080, 0x0000800041000080,
    0x00FFFCDDFCED714A, 0x007FFCDDFCED714A, 0x003FFFCDFFD88096, 0x0000040810002101,
    0x0001000204080011, 0x0001000204000801, 0x0001000082000401, 0x0001FFFAABFAD1A2,
];

/// Rook magic entries for each square.
pub static ROOK_MAGICS: [Magic; 64] = init_rook_magics();

/// Precomputed rook attack table (all occupancy configurations).
static ROOK_ATTACKS: [Bitboard; 102400] = init_rook_attacks();

/// Get rook attacks for a given square and occupancy.
#[inline(always)]
pub fn rook_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let magic = &ROOK_MAGICS[sq.index() as usize];
    ROOK_ATTACKS[magic.index(occ)]
}

/// Initialize rook magic entries (const-compatible).
const fn init_rook_magics() -> [Magic; 64] {
    let mut magics = [Magic::EMPTY; 64];
    let mut offset = 0usize;
    
    let mut sq = 0u8;
    while sq < 64 {
        let square = unsafe { Square::from_index_unchecked(sq) };
        let mask = rook_mask(square);
        let bits = mask.count() as u8;
        let shift = 64 - bits;
        let size = 1usize << bits;
        
        magics[sq as usize] = Magic {
            mask,
            magic: ROOK_MAGIC_NUMBERS[sq as usize],
            offset,
            shift,
        };
        
        offset += size;
        sq += 1;
    }
    
    magics
}

/// Initialize rook attack table.
const fn init_rook_attacks() -> [Bitboard; 102400] {
    let mut attacks = [Bitboard::EMPTY; 102400];
    
    let mut sq = 0u8;
    let mut offset = 0usize;
    
    while sq < 64 {
        let square = unsafe { Square::from_index_unchecked(sq) };
        let mask = rook_mask(square);
        let magic = ROOK_MAGIC_NUMBERS[sq as usize];
        let bits = mask.count() as u8;
        let shift = 64 - bits;
        let size = 1usize << bits;
        
        // Enumerate all occupancy configurations
        let mut idx = 0usize;
        while idx < size {
            // Reconstruct occupancy from index
            let mut occ = 0u64;
            let mut m = mask.0;
            let mut i = idx;
            while m != 0 {
                let lsb = m & m.wrapping_neg();
                if (i & 1) != 0 {
                    occ |= lsb;
                }
                m &= m - 1;
                i >>= 1;
            }
            
            // Compute attacks
            let attack = rook_attacks_slow(square, Bitboard(occ));
            
            // Compute index
            let hash = occ.wrapping_mul(magic);
            let index = offset + ((hash >> shift) as usize);
            
            attacks[index] = attack;
            idx += 1;
        }
        
        offset += size;
        sq += 1;
    }
    
    attacks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rook_attacks_e4_empty() {
        let attacks = rook_attacks(Square::E4, Bitboard::EMPTY);
        // E-file (7 squares) + 4th rank (7 squares) = 14
        assert_eq!(attacks.count(), 14);
    }

    #[test]
    fn test_rook_attacks_with_blocker() {
        let occ = Bitboard::from_square(Square::E6);
        let attacks = rook_attacks(Square::E4, occ);
        assert!(attacks.contains(Square::E5));
        assert!(attacks.contains(Square::E6));  // Can capture blocker
        assert!(!attacks.contains(Square::E7)); // Blocked
    }

    #[test]
    fn test_rook_attacks_corner() {
        let attacks = rook_attacks(Square::A1, Bitboard::EMPTY);
        assert_eq!(attacks.count(), 14);
        assert!(attacks.contains(Square::A8));
        assert!(attacks.contains(Square::H1));
    }
}
