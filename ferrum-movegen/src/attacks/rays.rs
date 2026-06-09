//! Ray tables for sliding piece move generation.

use crate::bitboard::Bitboard;
use crate::types::{Square, Direction};

/// Precomputed ray tables [direction][square].
/// Each ray extends from the square in the given direction until the board edge.
pub static RAYS: [[Bitboard; 64]; 8] = generate_rays();

/// Get the ray from a square in a given direction.
#[inline(always)]
pub fn ray(dir: Direction, sq: Square) -> Bitboard {
    let idx = match dir {
        Direction::North => 0,
        Direction::South => 1,
        Direction::East => 2,
        Direction::West => 3,
        Direction::NorthEast => 4,
        Direction::NorthWest => 5,
        Direction::SouthEast => 6,
        Direction::SouthWest => 7,
    };
    RAYS[idx][sq.index() as usize]
}

/// Generate ray tables at compile time.
const fn generate_rays() -> [[Bitboard; 64]; 8] {
    let mut rays = [[Bitboard::EMPTY; 64]; 8];
    
    let mut sq = 0u8;
    while sq < 64 {
        let file = sq & 7;
        let rank = sq >> 3;
        
        // North ray
        {
            let mut ray = 0u64;
            let mut r = rank + 1;
            while r < 8 {
                ray |= 1u64 << (r * 8 + file);
                r += 1;
            }
            rays[0][sq as usize] = Bitboard(ray);
        }
        
        // South ray
        {
            let mut ray = 0u64;
            let mut r = rank;
            while r > 0 {
                r -= 1;
                ray |= 1u64 << (r * 8 + file);
            }
            rays[1][sq as usize] = Bitboard(ray);
        }
        
        // East ray
        {
            let mut ray = 0u64;
            let mut f = file + 1;
            while f < 8 {
                ray |= 1u64 << (rank * 8 + f);
                f += 1;
            }
            rays[2][sq as usize] = Bitboard(ray);
        }
        
        // West ray
        {
            let mut ray = 0u64;
            let mut f = file;
            while f > 0 {
                f -= 1;
                ray |= 1u64 << (rank * 8 + f);
            }
            rays[3][sq as usize] = Bitboard(ray);
        }
        
        // Northeast ray
        {
            let mut ray = 0u64;
            let mut r = rank + 1;
            let mut f = file + 1;
            while r < 8 && f < 8 {
                ray |= 1u64 << (r * 8 + f);
                r += 1;
                f += 1;
            }
            rays[4][sq as usize] = Bitboard(ray);
        }
        
        // Northwest ray
        {
            let mut ray = 0u64;
            let mut r = rank + 1;
            let mut f = file;
            while r < 8 && f > 0 {
                f -= 1;
                ray |= 1u64 << (r * 8 + f);
                r += 1;
            }
            rays[5][sq as usize] = Bitboard(ray);
        }
        
        // Southeast ray
        {
            let mut ray = 0u64;
            let mut r = rank;
            let mut f = file + 1;
            while r > 0 && f < 8 {
                r -= 1;
                ray |= 1u64 << (r * 8 + f);
                f += 1;
            }
            rays[6][sq as usize] = Bitboard(ray);
        }
        
        // Southwest ray
        {
            let mut ray = 0u64;
            let mut r = rank;
            let mut f = file;
            while r > 0 && f > 0 {
                r -= 1;
                f -= 1;
                ray |= 1u64 << (r * 8 + f);
            }
            rays[7][sq as usize] = Bitboard(ray);
        }
        
        sq += 1;
    }
    
    rays
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ray_north() {
        let r = ray(Direction::North, Square::E4);
        assert!(r.contains(Square::E5));
        assert!(r.contains(Square::E6));
        assert!(r.contains(Square::E7));
        assert!(r.contains(Square::E8));
        assert!(!r.contains(Square::E4));
        assert!(!r.contains(Square::E3));
    }

    #[test]
    fn test_ray_diagonal() {
        let r = ray(Direction::NorthEast, Square::A1);
        assert!(r.contains(Square::B2));
        assert!(r.contains(Square::H8));
        assert_eq!(r.count(), 7);
    }
}
