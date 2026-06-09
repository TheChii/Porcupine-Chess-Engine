//! Line and between bitboards for pin/check detection.

use crate::bitboard::Bitboard;
use crate::types::Square;

/// Precomputed line tables [square1][square2].
/// Contains the full line through both squares if they're on a rank, file, or diagonal.
static LINE: [[Bitboard; 64]; 64] = generate_lines();

/// Precomputed between tables [square1][square2].
/// Contains the squares strictly between the two squares (exclusive).
static BETWEEN: [[Bitboard; 64]; 64] = generate_between();

/// Get the full line through two squares.
/// Returns empty bitboard if squares are not on the same rank, file, or diagonal.
#[inline(always)]
pub fn line(sq1: Square, sq2: Square) -> Bitboard {
    LINE[sq1.index() as usize][sq2.index() as usize]
}

/// Get the squares between two squares (exclusive).
/// Returns empty bitboard if squares are not on the same rank, file, or diagonal.
#[inline(always)]
pub fn between(sq1: Square, sq2: Square) -> Bitboard {
    BETWEEN[sq1.index() as usize][sq2.index() as usize]
}

/// Generate line tables at compile time.
const fn generate_lines() -> [[Bitboard; 64]; 64] {
    let mut lines = [[Bitboard::EMPTY; 64]; 64];
    
    let mut sq1 = 0u8;
    while sq1 < 64 {
        let mut sq2 = 0u8;
        while sq2 < 64 {
            let f1 = sq1 & 7;
            let r1 = sq1 >> 3;
            let f2 = sq2 & 7;
            let r2 = sq2 >> 3;
            
            let df = (f2 as i8) - (f1 as i8);
            let dr = (r2 as i8) - (r1 as i8);
            
            // Check if on same line
            let on_line = if df == 0 && dr != 0 {
                true // Same file
            } else if dr == 0 && df != 0 {
                true // Same rank
            } else if df.abs() == dr.abs() && df != 0 {
                true // Diagonal
            } else {
                false
            };
            
            if on_line {
                let mut line_bb = 0u64;
                
                // Normalize direction
                let df_norm = if df == 0 { 0i8 } else if df > 0 { 1 } else { -1 };
                let dr_norm = if dr == 0 { 0i8 } else if dr > 0 { 1 } else { -1 };
                let delta = dr_norm * 8 + df_norm;
                
                // Go to one end of the line
                let mut pos = sq1 as i8;
                while pos >= 0 && pos < 64 {
                    let f = pos & 7;
                    let r = pos >> 3;
                    let new_f = f - df_norm;
                    let new_r = r - dr_norm;
                    
                    if new_f < 0 || new_f > 7 || new_r < 0 || new_r > 7 {
                        break;
                    }
                    pos -= delta;
                }
                
                // Walk the entire line
                while pos >= 0 && pos < 64 {
                    line_bb |= 1u64 << pos;
                    
                    let f = pos & 7;
                    let r = pos >> 3;
                    let new_f = f + df_norm;
                    let new_r = r + dr_norm;
                    
                    if new_f < 0 || new_f > 7 || new_r < 0 || new_r > 7 {
                        break;
                    }
                    pos += delta;
                }
                
                lines[sq1 as usize][sq2 as usize] = Bitboard(line_bb);
            }
            
            sq2 += 1;
        }
        sq1 += 1;
    }
    
    lines
}

/// Generate between tables at compile time.
const fn generate_between() -> [[Bitboard; 64]; 64] {
    let mut between = [[Bitboard::EMPTY; 64]; 64];
    
    let mut sq1 = 0u8;
    while sq1 < 64 {
        let mut sq2 = 0u8;
        while sq2 < 64 {
            if sq1 == sq2 {
                sq2 += 1;
                continue;
            }
            
            let f1 = sq1 & 7;
            let r1 = sq1 >> 3;
            let f2 = sq2 & 7;
            let r2 = sq2 >> 3;
            
            let df = (f2 as i8) - (f1 as i8);
            let dr = (r2 as i8) - (r1 as i8);
            
            // Check if on same line
            let on_line = if df == 0 && dr != 0 {
                true
            } else if dr == 0 && df != 0 {
                true
            } else if df.abs() == dr.abs() {
                true
            } else {
                false
            };
            
            if on_line {
                let mut between_bb = 0u64;
                
                let df_norm = if df == 0 { 0i8 } else if df > 0 { 1 } else { -1 };
                let dr_norm = if dr == 0 { 0i8 } else if dr > 0 { 1 } else { -1 };
                let delta = dr_norm * 8 + df_norm;
                
                let mut pos = (sq1 as i8) + delta;
                while pos != sq2 as i8 && pos >= 0 && pos < 64 {
                    between_bb |= 1u64 << pos;
                    pos += delta;
                }
                
                between[sq1 as usize][sq2 as usize] = Bitboard(between_bb);
            }
            
            sq2 += 1;
        }
        sq1 += 1;
    }
    
    between
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_between_file() {
        // Between a1 and a8
        let b = between(Square::A1, Square::A8);
        assert_eq!(b.count(), 6); // a2-a7
        assert!(b.contains(Square::A2));
        assert!(b.contains(Square::A7));
        assert!(!b.contains(Square::A1));
        assert!(!b.contains(Square::A8));
    }

    #[test]
    fn test_between_rank() {
        // Between a4 and h4
        let b = between(Square::A4, Square::H4);
        assert_eq!(b.count(), 6); // b4-g4
    }

    #[test]
    fn test_between_diagonal() {
        // Between a1 and h8
        let b = between(Square::A1, Square::H8);
        assert_eq!(b.count(), 6); // b2-g7
        assert!(b.contains(Square::D4));
    }

    #[test]
    fn test_line() {
        // Line through e4 and e6 includes all of e-file
        let l = line(Square::E4, Square::E6);
        assert!(l.contains(Square::E1));
        assert!(l.contains(Square::E8));
        assert_eq!(l.count(), 8);
    }

    #[test]
    fn test_no_line() {
        // No line between a1 and b3 (knight move)
        let l = line(Square::A1, Square::B3);
        assert!(l.is_empty());
    }
}
