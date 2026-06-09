//! King attack tables.

use crate::bitboard::Bitboard;
use crate::types::Square;

/// Precomputed king attack table [square].
pub static KING_ATTACKS: [Bitboard; 64] = generate_king_attacks();

/// Get king attacks for a given square.
#[inline(always)]
pub fn king_attacks(sq: Square) -> Bitboard {
    KING_ATTACKS[sq.index() as usize]
}

/// Generate king attack table at compile time.
const fn generate_king_attacks() -> [Bitboard; 64] {
    let mut attacks = [Bitboard::EMPTY; 64];
    
    let mut sq = 0u8;
    while sq < 64 {
        let bb = 1u64 << sq;
        let mut attack = 0u64;
        
        // King moves: 1 square in each direction
        // North
        attack |= bb << 8;
        // South
        attack |= bb >> 8;
        // East (not from h-file)
        attack |= (bb << 1) & !Bitboard::FILE_A.0;
        // West (not from a-file)
        attack |= (bb >> 1) & !Bitboard::FILE_H.0;
        // Northeast
        attack |= (bb << 9) & !Bitboard::FILE_A.0;
        // Northwest
        attack |= (bb << 7) & !Bitboard::FILE_H.0;
        // Southeast
        attack |= (bb >> 7) & !Bitboard::FILE_A.0;
        // Southwest
        attack |= (bb >> 9) & !Bitboard::FILE_H.0;
        
        attacks[sq as usize] = Bitboard(attack);
        sq += 1;
    }
    
    attacks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_king_center() {
        // King on e4 attacks 8 squares
        let attacks = king_attacks(Square::E4);
        assert_eq!(attacks.count(), 8);
        assert!(attacks.contains(Square::D3));
        assert!(attacks.contains(Square::E3));
        assert!(attacks.contains(Square::F3));
        assert!(attacks.contains(Square::D4));
        assert!(attacks.contains(Square::F4));
        assert!(attacks.contains(Square::D5));
        assert!(attacks.contains(Square::E5));
        assert!(attacks.contains(Square::F5));
    }

    #[test]
    fn test_king_corner() {
        // King on a1 attacks only 3 squares
        let attacks = king_attacks(Square::A1);
        assert_eq!(attacks.count(), 3);
        assert!(attacks.contains(Square::A2));
        assert!(attacks.contains(Square::B1));
        assert!(attacks.contains(Square::B2));
    }

    #[test]
    fn test_king_edge() {
        // King on a4 attacks 5 squares
        let attacks = king_attacks(Square::A4);
        assert_eq!(attacks.count(), 5);
    }
}
