//! Zobrist hashing for board positions.

use crate::types::{Square, Piece, Color, File, CastleRights};

/// Zobrist hash keys.
pub struct Zobrist {
    /// Piece-square keys [piece][color][square].
    piece_squares: [[[u64; 64]; 2]; 6],
    /// Side to move key.
    side: u64,
    /// Castling rights keys [16 possible combinations].
    castling: [u64; 16],
    /// En passant file keys.
    ep_file: [u64; 8],
}

impl Zobrist {
    /// Generate Zobrist keys using a simple PRNG.
    const fn new() -> Zobrist {
        let mut piece_squares = [[[0u64; 64]; 2]; 6];
        let side;
        let mut castling = [0u64; 16];
        let mut ep_file = [0u64; 8];
        
        // Simple xorshift PRNG
        let mut state = 0x3243F6A8885A308Du64; // PI digits
        
        // Helper to get next random
        macro_rules! next_rand {
            ($state:expr) => {{
                $state ^= $state << 13;
                $state ^= $state >> 7;
                $state ^= $state << 17;
                $state
            }};
        }
        
        // Generate piece-square keys
        let mut piece = 0;
        while piece < 6 {
            let mut color = 0;
            while color < 2 {
                let mut sq = 0;
                while sq < 64 {
                    piece_squares[piece][color][sq] = next_rand!(state);
                    sq += 1;
                }
                color += 1;
            }
            piece += 1;
        }
        
        // Side to move
        side = next_rand!(state);
        
        // Castling keys
        let mut i = 0;
        while i < 16 {
            castling[i] = next_rand!(state);
            i += 1;
        }
        
        // EP file keys
        let mut i = 0;
        while i < 8 {
            ep_file[i] = next_rand!(state);
            i += 1;
        }
        
        Zobrist {
            piece_squares,
            side,
            castling,
            ep_file,
        }
    }

    /// Get piece-square key.
    #[inline(always)]
    pub const fn piece_square(&self, piece: Piece, color: Color, sq: Square) -> u64 {
        self.piece_squares[piece as usize][color as usize][sq.index() as usize]
    }

    /// Get side-to-move key.
    #[inline(always)]
    pub const fn side(&self) -> u64 {
        self.side
    }

    /// Get castling rights key.
    #[inline(always)]
    pub const fn castling(&self, rights: CastleRights) -> u64 {
        self.castling[rights.bits() as usize]
    }

    /// Get en passant file key.
    #[inline(always)]
    pub const fn ep_file(&self, file: File) -> u64 {
        self.ep_file[file.index() as usize]
    }
}

/// Global Zobrist keys instance.
pub static ZOBRIST: Zobrist = Zobrist::new();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zobrist_unique() {
        // Check that different piece-square combinations give different keys
        let k1 = ZOBRIST.piece_square(Piece::Pawn, Color::White, Square::E2);
        let k2 = ZOBRIST.piece_square(Piece::Pawn, Color::White, Square::E4);
        let k3 = ZOBRIST.piece_square(Piece::Pawn, Color::Black, Square::E2);
        
        assert_ne!(k1, k2);
        assert_ne!(k1, k3);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_zobrist_side() {
        assert_ne!(ZOBRIST.side(), 0);
    }
}
