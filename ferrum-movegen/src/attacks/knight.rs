//! Knight attack tables.

use crate::bitboard::Bitboard;
use crate::types::Square;

/// Precomputed knight attack table [square].
pub static KNIGHT_ATTACKS: [Bitboard; 64] = generate_knight_attacks();

/// Get knight attacks for a given square.
#[inline(always)]
pub fn knight_attacks(sq: Square) -> Bitboard {
    KNIGHT_ATTACKS[sq.index() as usize]
}

/// Generate knight attack table at compile time.
const fn generate_knight_attacks() -> [Bitboard; 64] {
    let mut attacks = [Bitboard::EMPTY; 64];
    
    let mut sq = 0u8;
    while sq < 64 {
        let bb = 1u64 << sq;
        let mut attack = 0u64;
        
        // Knight moves: 2+1 or 1+2 in each direction
        // For 1-file moves (up/down 2, left/right 1): mask destination to avoid wrap
        // For 2-file moves (up/down 1, left/right 2): mask source to avoid wrap
        
        // NNE: +17 (up 2, right 1) - mask destination (exclude file A)
        attack |= (bb << 17) & !Bitboard::FILE_A.0;
        // NNW: +15 (up 2, left 1) - mask destination (exclude file H)
        attack |= (bb << 15) & !Bitboard::FILE_H.0;
        // NEE: +10 (up 1, right 2) - mask source (exclude files G,H)
        attack |= (bb & Bitboard::NOT_FILE_GH.0) << 10;
        // NWW: +6 (up 1, left 2) - mask source (exclude files A,B)
        attack |= (bb & Bitboard::NOT_FILE_AB.0) << 6;
        // SSE: -15 (down 2, right 1) - mask destination (exclude file A)
        attack |= (bb >> 15) & !Bitboard::FILE_A.0;
        // SSW: -17 (down 2, left 1) - mask destination (exclude file H)
        attack |= (bb >> 17) & !Bitboard::FILE_H.0;
        // SEE: -6 (down 1, right 2) - mask source (exclude files G,H)
        attack |= (bb & Bitboard::NOT_FILE_GH.0) >> 6;
        // SWW: -10 (down 1, left 2) - mask source (exclude files A,B)
        attack |= (bb & Bitboard::NOT_FILE_AB.0) >> 10;
        
        attacks[sq as usize] = Bitboard(attack);
        sq += 1;
    }
    
    attacks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knight_center() {
        // Knight on e4 attacks 8 squares
        let attacks = knight_attacks(Square::E4);
        assert_eq!(attacks.count(), 8);
        assert!(attacks.contains(Square::D6));
        assert!(attacks.contains(Square::F6));
        assert!(attacks.contains(Square::G5));
        assert!(attacks.contains(Square::G3));
        assert!(attacks.contains(Square::F2));
        assert!(attacks.contains(Square::D2));
        assert!(attacks.contains(Square::C3));
        assert!(attacks.contains(Square::C5));
    }

    #[test]
    fn test_knight_corner() {
        // Knight on a1 attacks only 2 squares
        let attacks = knight_attacks(Square::A1);
        assert_eq!(attacks.count(), 2);
        assert!(attacks.contains(Square::B3));
        assert!(attacks.contains(Square::C2));
    }

    #[test]
    fn test_knight_edge() {
        // Knight on h4 attacks fewer squares
        let attacks = knight_attacks(Square::H4);
        assert_eq!(attacks.count(), 4);
    }
}
