//! Bishop magic bitboards.

use super::{Magic, bishop_mask, bishop_attacks_slow};
use crate::bitboard::Bitboard;
use crate::types::Square;

/// Bishop magic numbers (precomputed).
const BISHOP_MAGIC_NUMBERS: [u64; 64] = [
    0x0002020202020200, 0x0002020202020000, 0x0004010202000000, 0x0004040080000000,
    0x0001104000000000, 0x0000821040000000, 0x0000410410400000, 0x0000104104104000,
    0x0000040404040400, 0x0000020202020200, 0x0000040102020000, 0x0000040400800000,
    0x0000011040000000, 0x0000008210400000, 0x0000004104104000, 0x0000002082082000,
    0x0004000808080800, 0x0002000404040400, 0x0001000202020200, 0x0000800802004000,
    0x0000800400A00000, 0x0000200100884000, 0x0000400082082000, 0x0000200041041000,
    0x0002080010101000, 0x0001040008080800, 0x0000208004010400, 0x0000404004010200,
    0x0000840000802000, 0x0000404002011000, 0x0000808001041000, 0x0000404000820800,
    0x0001041000202000, 0x0000820800101000, 0x0000104400080800, 0x0000020080080080,
    0x0000404040040100, 0x0000808100020100, 0x0001010100020800, 0x0000808080010400,
    0x0000820820004000, 0x0000410410002000, 0x0000082088001000, 0x0000002011000800,
    0x0000080100400400, 0x0001010101000200, 0x0002020202000400, 0x0001010101000200,
    0x0000410410400000, 0x0000208208200000, 0x0000002084100000, 0x0000000020880000,
    0x0000001002020000, 0x0000040408020000, 0x0004040404040000, 0x0002020202020000,
    0x0000104104104000, 0x0000002082082000, 0x0000000020841000, 0x0000000000208800,
    0x0000000010020200, 0x0000000404080200, 0x0000040404040400, 0x0002020202020200,
];

/// Bishop magic entries for each square.
pub static BISHOP_MAGICS: [Magic; 64] = init_bishop_magics();

/// Precomputed bishop attack table.
static BISHOP_ATTACKS: [Bitboard; 5248] = init_bishop_attacks();

/// Get bishop attacks for a given square and occupancy.
#[inline(always)]
pub fn bishop_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let magic = &BISHOP_MAGICS[sq.index() as usize];
    BISHOP_ATTACKS[magic.index(occ)]
}

/// Initialize bishop magic entries.
const fn init_bishop_magics() -> [Magic; 64] {
    let mut magics = [Magic::EMPTY; 64];
    let mut offset = 0usize;
    
    let mut sq = 0u8;
    while sq < 64 {
        let square = unsafe { Square::from_index_unchecked(sq) };
        let mask = bishop_mask(square);
        let bits = mask.count() as u8;
        let shift = 64 - bits;
        let size = 1usize << bits;
        
        magics[sq as usize] = Magic {
            mask,
            magic: BISHOP_MAGIC_NUMBERS[sq as usize],
            offset,
            shift,
        };
        
        offset += size;
        sq += 1;
    }
    
    magics
}

/// Initialize bishop attack table.
const fn init_bishop_attacks() -> [Bitboard; 5248] {
    let mut attacks = [Bitboard::EMPTY; 5248];
    
    let mut sq = 0u8;
    let mut offset = 0usize;
    
    while sq < 64 {
        let square = unsafe { Square::from_index_unchecked(sq) };
        let mask = bishop_mask(square);
        let magic = BISHOP_MAGIC_NUMBERS[sq as usize];
        let bits = mask.count() as u8;
        let shift = 64 - bits;
        let size = 1usize << bits;
        
        let mut idx = 0usize;
        while idx < size {
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
            
            let attack = bishop_attacks_slow(square, Bitboard(occ));
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
    fn test_bishop_attacks_e4_empty() {
        let attacks = bishop_attacks(Square::E4, Bitboard::EMPTY);
        // a8-h1 diagonal + a2-f7 diagonal minus e4 itself
        assert_eq!(attacks.count(), 13);
    }

    #[test]
    fn test_bishop_attacks_with_blocker() {
        let occ = Bitboard::from_square(Square::G6);
        let attacks = bishop_attacks(Square::E4, occ);
        assert!(attacks.contains(Square::F5));
        assert!(attacks.contains(Square::G6));  // Can capture
        assert!(!attacks.contains(Square::H7)); // Blocked
    }

    #[test]
    fn test_bishop_attacks_corner() {
        let attacks = bishop_attacks(Square::A1, Bitboard::EMPTY);
        assert_eq!(attacks.count(), 7);
        assert!(attacks.contains(Square::H8));
    }
}
