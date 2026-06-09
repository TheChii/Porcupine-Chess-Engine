//! Pawn attack tables.

use crate::bitboard::Bitboard;
use crate::types::{Color, Square};

/// Precomputed pawn attack tables [color][square].
pub static PAWN_ATTACKS: [[Bitboard; 64]; 2] = generate_pawn_attacks();

/// Get pawn attacks for a given color and square.
#[inline(always)]
pub fn pawn_attacks(color: Color, sq: Square) -> Bitboard {
    PAWN_ATTACKS[color.index()][sq.index() as usize]
}

/// Generate all pawn attack tables at compile time.
const fn generate_pawn_attacks() -> [[Bitboard; 64]; 2] {
    let mut attacks = [[Bitboard::EMPTY; 64]; 2];
    
    let mut sq = 0u8;
    while sq < 64 {
        let bb = 1u64 << sq;
        
        // White pawn attacks (northeast and northwest)
        let white_ne = (bb << 9) & !Bitboard::FILE_A.0;
        let white_nw = (bb << 7) & !Bitboard::FILE_H.0;
        attacks[0][sq as usize] = Bitboard(white_ne | white_nw);
        
        // Black pawn attacks (southeast and southwest)
        let black_se = (bb >> 7) & !Bitboard::FILE_A.0;
        let black_sw = (bb >> 9) & !Bitboard::FILE_H.0;
        attacks[1][sq as usize] = Bitboard(black_se | black_sw);
        
        sq += 1;
    }
    
    attacks
}

/// Get all pawn attacks for a bitboard of pawns.
#[inline]
pub fn pawn_attacks_bb(color: Color, pawns: Bitboard) -> Bitboard {
    match color {
        Color::White => pawns.north_east() | pawns.north_west(),
        Color::Black => pawns.south_east() | pawns.south_west(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_white_pawn_attacks() {
        // e4 pawn attacks d5 and f5
        let attacks = pawn_attacks(Color::White, Square::E4);
        assert!(attacks.contains(Square::D5));
        assert!(attacks.contains(Square::F5));
        assert_eq!(attacks.count(), 2);
    }

    #[test]
    fn test_black_pawn_attacks() {
        // e5 pawn attacks d4 and f4
        let attacks = pawn_attacks(Color::Black, Square::E5);
        assert!(attacks.contains(Square::D4));
        assert!(attacks.contains(Square::F4));
        assert_eq!(attacks.count(), 2);
    }

    #[test]
    fn test_edge_pawns() {
        // a4 white pawn only attacks b5
        let attacks = pawn_attacks(Color::White, Square::A4);
        assert_eq!(attacks.count(), 1);
        assert!(attacks.contains(Square::B5));

        // h4 white pawn only attacks g5
        let attacks = pawn_attacks(Color::White, Square::H4);
        assert_eq!(attacks.count(), 1);
        assert!(attacks.contains(Square::G5));
    }
}
