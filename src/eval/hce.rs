//! Unified Hand-Crafted Evaluation (HCE) - Ultra Optimized
//!
//! Single evaluation function for all game phases with:
//! - Packed MG/EG scores for efficient tapered evaluation
//! - Precomputed distance tables (compile-time)
//! - Branchless arithmetic via const generics
//! - Cache-aligned PST arrays
//! - Endgame-aware bonuses (king proximity, passed pawns, corner driving)
//!
//! Used as NNUE fallback and works seamlessly across all phases.

use crate::types::{Board, Score, Color, Piece, Bitboard};
use movegen::Square;

// ============================================================================
// PACKED SCORE TYPE
// ============================================================================

/// Packed middlegame/endgame score.
/// Uses separate i16 values for simplicity and correctness.
/// This avoids overflow issues with the packed single-i32 approach.
#[derive(Clone, Copy, Default)]
struct S {
    mg: i16,
    eg: i16,
}

impl S {
    #[inline(always)]
    const fn new(mg: i16, eg: i16) -> Self {
        Self { mg, eg }
    }

    #[inline(always)]
    const fn mg(self) -> i32 {
        self.mg as i32
    }

    #[inline(always)]
    const fn eg(self) -> i32 {
        self.eg as i32
    }
}

impl core::ops::Add for S {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            mg: self.mg.wrapping_add(rhs.mg),
            eg: self.eg.wrapping_add(rhs.eg),
        }
    }
}

impl core::ops::Sub for S {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            mg: self.mg.wrapping_sub(rhs.mg),
            eg: self.eg.wrapping_sub(rhs.eg),
        }
    }
}

impl core::ops::AddAssign for S {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.mg = self.mg.wrapping_add(rhs.mg);
        self.eg = self.eg.wrapping_add(rhs.eg);
    }
}

impl core::ops::SubAssign for S {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.mg = self.mg.wrapping_sub(rhs.mg);
        self.eg = self.eg.wrapping_sub(rhs.eg);
    }
}

impl core::ops::Neg for S {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            mg: self.mg.wrapping_neg(),
            eg: self.eg.wrapping_neg(),
        }
    }
}

// ============================================================================
// PIECE VALUES (Packed MG/EG)
// ============================================================================

const PIECE_VALUES: [S; 6] = [
    S::new(100, 120),   // Pawn
    S::new(320, 300),   // Knight  
    S::new(330, 320),   // Bishop
    S::new(500, 550),   // Rook
    S::new(950, 1000),  // Queen
    S::new(0, 0),       // King (no material value)
];

const BISHOP_PAIR: S = S::new(35, 50);

// ============================================================================
// PIECE-SQUARE TABLES (Packed MG/EG, White's perspective, A1=0)
// ============================================================================

/// PST indexed by [piece][square], packed MG/EG
#[repr(align(64))]
struct PstTable([[S; 64]; 6]);

static PST: PstTable = PstTable([
    // Pawn
    [
        S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0),
        S::new(5, 5), S::new(10, 5), S::new(10, 5), S::new(-20, 5), S::new(-20, 5), S::new(10, 5), S::new(10, 5), S::new(5, 5),
        S::new(5, 10), S::new(-5, 10), S::new(-10, 10), S::new(0, 10), S::new(0, 10), S::new(-10, 10), S::new(-5, 10), S::new(5, 10),
        S::new(0, 25), S::new(0, 25), S::new(0, 25), S::new(20, 25), S::new(20, 25), S::new(0, 25), S::new(0, 25), S::new(0, 25),
        S::new(5, 40), S::new(5, 40), S::new(10, 40), S::new(25, 40), S::new(25, 40), S::new(10, 40), S::new(5, 40), S::new(5, 40),
        S::new(10, 60), S::new(10, 60), S::new(20, 60), S::new(30, 60), S::new(30, 60), S::new(20, 60), S::new(10, 60), S::new(10, 60),
        S::new(50, 100), S::new(50, 100), S::new(50, 100), S::new(50, 100), S::new(50, 100), S::new(50, 100), S::new(50, 100), S::new(50, 100),
        S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0),
    ],
    // Knight
    [
        S::new(-50, -50), S::new(-40, -40), S::new(-30, -30), S::new(-30, -30), S::new(-30, -30), S::new(-30, -30), S::new(-40, -40), S::new(-50, -50),
        S::new(-40, -40), S::new(-20, -20), S::new(0, 0), S::new(5, 5), S::new(5, 5), S::new(0, 0), S::new(-20, -20), S::new(-40, -40),
        S::new(-30, -30), S::new(5, 5), S::new(10, 10), S::new(15, 15), S::new(15, 15), S::new(10, 10), S::new(5, 5), S::new(-30, -30),
        S::new(-30, -30), S::new(0, 0), S::new(15, 15), S::new(20, 20), S::new(20, 20), S::new(15, 15), S::new(0, 0), S::new(-30, -30),
        S::new(-30, -30), S::new(5, 5), S::new(15, 15), S::new(20, 20), S::new(20, 20), S::new(15, 15), S::new(5, 5), S::new(-30, -30),
        S::new(-30, -30), S::new(0, 0), S::new(10, 10), S::new(15, 15), S::new(15, 15), S::new(10, 10), S::new(0, 0), S::new(-30, -30),
        S::new(-40, -40), S::new(-20, -20), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(-20, -20), S::new(-40, -40),
        S::new(-50, -50), S::new(-40, -40), S::new(-30, -30), S::new(-30, -30), S::new(-30, -30), S::new(-30, -30), S::new(-40, -40), S::new(-50, -50),
    ],
    // Bishop
    [
        S::new(-20, -20), S::new(-10, -10), S::new(-10, -10), S::new(-10, -10), S::new(-10, -10), S::new(-10, -10), S::new(-10, -10), S::new(-20, -20),
        S::new(-10, -10), S::new(5, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(5, 0), S::new(-10, -10),
        S::new(-10, -10), S::new(10, 10), S::new(10, 10), S::new(10, 10), S::new(10, 10), S::new(10, 10), S::new(10, 10), S::new(-10, -10),
        S::new(-10, -10), S::new(0, 0), S::new(10, 10), S::new(10, 10), S::new(10, 10), S::new(10, 10), S::new(0, 0), S::new(-10, -10),
        S::new(-10, -10), S::new(5, 5), S::new(5, 5), S::new(10, 10), S::new(10, 10), S::new(5, 5), S::new(5, 5), S::new(-10, -10),
        S::new(-10, -10), S::new(0, 0), S::new(5, 5), S::new(10, 10), S::new(10, 10), S::new(5, 5), S::new(0, 0), S::new(-10, -10),
        S::new(-10, -10), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(-10, -10),
        S::new(-20, -20), S::new(-10, -10), S::new(-10, -10), S::new(-10, -10), S::new(-10, -10), S::new(-10, -10), S::new(-10, -10), S::new(-20, -20),
    ],
    // Rook
    [
        S::new(0, 0), S::new(0, 0), S::new(0, 5), S::new(5, 5), S::new(5, 5), S::new(0, 5), S::new(0, 0), S::new(0, 0),
        S::new(-5, -5), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(-5, -5),
        S::new(-5, -5), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(-5, -5),
        S::new(-5, -5), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(-5, -5),
        S::new(-5, -5), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(-5, -5),
        S::new(-5, -5), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(-5, -5),
        S::new(5, 10), S::new(10, 10), S::new(10, 10), S::new(10, 10), S::new(10, 10), S::new(10, 10), S::new(10, 10), S::new(5, 10),
        S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0),
    ],
    // Queen
    [
        S::new(-20, -20), S::new(-10, -10), S::new(-10, -10), S::new(-5, -5), S::new(-5, -5), S::new(-10, -10), S::new(-10, -10), S::new(-20, -20),
        S::new(-10, -10), S::new(0, 0), S::new(5, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(-10, -10),
        S::new(-10, -10), S::new(5, 5), S::new(5, 5), S::new(5, 5), S::new(5, 5), S::new(5, 5), S::new(0, 5), S::new(-10, -10),
        S::new(0, -5), S::new(0, 0), S::new(5, 5), S::new(5, 5), S::new(5, 5), S::new(5, 5), S::new(0, 0), S::new(-5, -5),
        S::new(-5, -5), S::new(0, 0), S::new(5, 5), S::new(5, 5), S::new(5, 5), S::new(5, 5), S::new(0, 0), S::new(-5, -5),
        S::new(-10, -10), S::new(0, 0), S::new(5, 5), S::new(5, 5), S::new(5, 5), S::new(5, 5), S::new(0, 0), S::new(-10, -10),
        S::new(-10, -10), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(-10, -10),
        S::new(-20, -20), S::new(-10, -10), S::new(-10, -10), S::new(-5, -5), S::new(-5, -5), S::new(-10, -10), S::new(-10, -10), S::new(-20, -20),
    ],
    // King (MG: castle/stay safe, EG: centralize)
    [
        S::new(20, -50), S::new(30, -30), S::new(10, -30), S::new(0, -30), S::new(0, -30), S::new(10, -30), S::new(30, -30), S::new(20, -50),
        S::new(20, -30), S::new(20, -30), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(0, 0), S::new(20, -30), S::new(20, -30),
        S::new(-10, -30), S::new(-20, -10), S::new(-20, 20), S::new(-20, 30), S::new(-20, 30), S::new(-20, 20), S::new(-20, -10), S::new(-10, -30),
        S::new(-20, -30), S::new(-30, -10), S::new(-30, 30), S::new(-40, 40), S::new(-40, 40), S::new(-30, 30), S::new(-30, -10), S::new(-20, -30),
        S::new(-30, -30), S::new(-40, -10), S::new(-40, 30), S::new(-50, 40), S::new(-50, 40), S::new(-40, 30), S::new(-40, -10), S::new(-30, -30),
        S::new(-30, -30), S::new(-40, -10), S::new(-40, 20), S::new(-50, 30), S::new(-50, 30), S::new(-40, 20), S::new(-40, -10), S::new(-30, -30),
        S::new(-30, -30), S::new(-40, -20), S::new(-40, -10), S::new(-50, 0), S::new(-50, 0), S::new(-40, -10), S::new(-40, -20), S::new(-30, -30),
        S::new(-30, -50), S::new(-40, -40), S::new(-40, -30), S::new(-50, -20), S::new(-50, -20), S::new(-40, -30), S::new(-40, -40), S::new(-30, -50),
    ],
]);

// ============================================================================
// PRECOMPUTED DISTANCE TABLES (Compile-time)
// ============================================================================

/// Center distance for each square (0 = center, 4 = corner)
static CENTER_DIST: [i32; 64] = {
    let mut table = [0i32; 64];
    let mut sq = 0;
    while sq < 64 {
        let file = sq % 8;
        let rank = sq / 8;
        let fd = if file < 4 { 3 - file } else { file - 4 };
        let rd = if rank < 4 { 3 - rank } else { rank - 4 };
        table[sq] = (fd + rd) as i32;
        sq += 1;
    }
    table
};

/// Corner distance for each square (0 = corner, 6 = center)
static CORNER_DIST: [i32; 64] = {
    let mut table = [0i32; 64];
    let mut sq = 0;
    while sq < 64 {
        let file = sq % 8;
        let rank = sq / 8;
        let fd = if file < 4 { file } else { 7 - file };
        let rd = if rank < 4 { rank } else { 7 - rank };
        table[sq] = (fd + rd) as i32;
        sq += 1;
    }
    table
};

/// Chebyshev (king) distance between any two squares
static KING_DIST: [[i32; 64]; 64] = {
    let mut table = [[0i32; 64]; 64];
    let mut sq1 = 0usize;
    while sq1 < 64 {
        let f1 = (sq1 % 8) as i32;
        let r1 = (sq1 / 8) as i32;
        let mut sq2 = 0usize;
        while sq2 < 64 {
            let f2 = (sq2 % 8) as i32;
            let r2 = (sq2 / 8) as i32;
            let fd = if f1 > f2 { f1 - f2 } else { f2 - f1 };
            let rd = if r1 > r2 { r1 - r2 } else { r2 - r1 };
            table[sq1][sq2] = if fd > rd { fd } else { rd };
            sq2 += 1;
        }
        sq1 += 1;
    }
    table
};

// ============================================================================
// PASSED PAWN MASKS (Precomputed)
// ============================================================================

/// Forward file + adjacent files mask for white pawns
static PASSED_MASK_WHITE: [u64; 64] = {
    let mut masks = [0u64; 64];
    let mut sq = 0usize;
    while sq < 64 {
        let file = sq % 8;
        let rank = sq / 8;
        let mut mask = 0u64;
        // For each rank above
        let mut r = rank + 1;
        while r < 8 {
            // Same file
            mask |= 1u64 << (r * 8 + file);
            // Left file
            if file > 0 {
                mask |= 1u64 << (r * 8 + file - 1);
            }
            // Right file
            if file < 7 {
                mask |= 1u64 << (r * 8 + file + 1);
            }
            r += 1;
        }
        masks[sq] = mask;
        sq += 1;
    }
    masks
};

/// Forward file + adjacent files mask for black pawns  
static PASSED_MASK_BLACK: [u64; 64] = {
    let mut masks = [0u64; 64];
    let mut sq = 0usize;
    while sq < 64 {
        let file = sq % 8;
        let rank = sq / 8;
        let mut mask = 0u64;
        // For each rank below
        if rank > 0 {
            let mut r = rank - 1;
            loop {
                // Same file
                mask |= 1u64 << (r * 8 + file);
                // Left file
                if file > 0 {
                    mask |= 1u64 << (r * 8 + file - 1);
                }
                // Right file
                if file < 7 {
                    mask |= 1u64 << (r * 8 + file + 1);
                }
                if r == 0 { break; }
                r -= 1;
            }
        }
        masks[sq] = mask;
        sq += 1;
    }
    masks
};

/// Passed pawn bonus by rank advancement (index 0-7)
static PASSED_BONUS: [S; 8] = [
    S::new(0, 0),       // Rank 1 (impossible for white)
    S::new(5, 10),      // Rank 2
    S::new(10, 20),     // Rank 3
    S::new(20, 40),     // Rank 4
    S::new(40, 70),     // Rank 5
    S::new(70, 120),    // Rank 6
    S::new(120, 200),   // Rank 7 (about to promote!)
    S::new(0, 0),       // Rank 8 (impossible)
];

// ============================================================================
// PHASE CALCULATION
// ============================================================================

const PHASE_TOTAL: i32 = 24; // 4*1 (N) + 4*1 (B) + 4*2 (R) + 2*4 (Q)

/// Calculate game phase (0 = endgame, 256 = opening)
#[inline(always)]
fn calculate_phase(board: &Board) -> i32 {
    let n = board.piece_bb(Piece::Knight).count() as i32;
    let b = board.piece_bb(Piece::Bishop).count() as i32;
    let r = board.piece_bb(Piece::Rook).count() as i32;
    let q = board.piece_bb(Piece::Queen).count() as i32;
    
    let material = n + b + 2 * r + 4 * q;
    
    // Clamp and scale to 0-256
    let phase = PHASE_TOTAL - material;
    if phase < 0 { 0 } else if phase > PHASE_TOTAL { 256 } else { (phase * 256) / PHASE_TOTAL }
}

// ============================================================================
// MAIN EVALUATION
// ============================================================================

/// Main evaluation function - returns score from side-to-move perspective
#[inline]
pub fn evaluate(board: &Board) -> Score {
    let phase = calculate_phase(board);
    
    // Evaluate both sides
    let white_score = eval_side::<true>(board);
    let black_score = eval_side::<false>(board);
    
    // Net score from white's perspective
    let mut score = white_score - black_score;
    
    // Add endgame-specific bonuses (scaled by phase)
    if phase > 128 {
        score = score + endgame_bonuses(board, phase);
    }
    
    // Taper between MG and EG
    let mg = score.mg();
    let eg = score.eg();
    let tapered = (mg * (256 - phase) + eg * phase) / 256;
    
    // Return from side-to-move perspective
    if board.turn() == Color::White {
        Score::cp(tapered)
    } else {
        Score::cp(-tapered)
    }
}

/// Evaluate one side using const generic for branchless color handling
#[inline(always)]
fn eval_side<const IS_WHITE: bool>(board: &Board) -> S {
    let color = if IS_WHITE { Color::White } else { Color::Black };
    let mut score = S::default();
    
    // Material + PST in single pass
    for (piece_idx, &piece) in [Piece::Pawn, Piece::Knight, Piece::Bishop, 
                                  Piece::Rook, Piece::Queen, Piece::King].iter().enumerate() {
        let pieces = board.piece_bb(piece) & board.color_bb(color);
        
        for sq in pieces {
            // Flip square for black (a1 -> a8, etc.)
            let sq_idx = if IS_WHITE { 
                sq.index() as usize 
            } else { 
                sq.index() as usize ^ 56 
            };
            
            // Material (not for king)
            if piece_idx < 5 {
                score += PIECE_VALUES[piece_idx];
            }
            
            // PST bonus
            score += PST.0[piece_idx][sq_idx];
        }
    }
    
    // Bishop pair bonus
    let bishops = board.piece_bb(Piece::Bishop) & board.color_bb(color);
    if bishops.count() >= 2 {
        score += BISHOP_PAIR;
    }
    
    // Passed pawn evaluation
    score = score + eval_passed_pawns::<IS_WHITE>(board);
    
    score
}

/// Evaluate passed pawns for one side
#[inline(always)]
fn eval_passed_pawns<const IS_WHITE: bool>(board: &Board) -> S {
    let color = if IS_WHITE { Color::White } else { Color::Black };
    let our_pawns = board.piece_bb(Piece::Pawn) & board.color_bb(color);
    let enemy_pawns = board.piece_bb(Piece::Pawn) & board.color_bb(!color);
    let enemy_pawns_bb = enemy_pawns.bits();
    
    let mut bonus = S::default();
    
    for sq in our_pawns {
        let sq_idx = sq.index() as usize;
        
        // Get appropriate passed pawn mask
        let mask = if IS_WHITE {
            PASSED_MASK_WHITE[sq_idx]
        } else {
            PASSED_MASK_BLACK[sq_idx]
        };
        
        // Check if passed (no enemy pawns in front or on adjacent files)
        if (enemy_pawns_bb & mask) == 0 {
            // Get rank advancement (0-7)
            let rank = if IS_WHITE {
                sq_idx / 8
            } else {
                7 - (sq_idx / 8)
            };
            
            bonus += PASSED_BONUS[rank];
        }
    }
    
    bonus
}

/// Endgame-specific bonuses (king proximity, corner driving)
#[inline(always)]
fn endgame_bonuses(board: &Board, phase: i32) -> S {
    // Only compute if actually in endgame-ish position
    let scale = (phase - 128) as i32; // 0-128 range
    
    // Determine if there's a material imbalance
    let material = material_balance(board);
    
    if material.abs() < 100 {
        // Roughly equal - no endgame bonuses
        return S::default();
    }
    
    let (winning_color, losing_color) = if material > 0 {
        (Color::White, Color::Black)
    } else {
        (Color::Black, Color::White)
    };
    
    let winner_king = board.king_square(winning_color);
    let loser_king = board.king_square(losing_color);
    
    let winner_sq = winner_king.index() as usize;
    let loser_sq = loser_king.index() as usize;
    
    // Push enemy king to corner (bonus for lower corner distance)
    let corner_bonus = (6 - CORNER_DIST[loser_sq]) * 8;
    
    // Push enemy king to edge (bonus for higher center distance)
    let edge_bonus = CENTER_DIST[loser_sq] * 6;
    
    // Bring our king closer
    let proximity_bonus = (7 - KING_DIST[winner_sq][loser_sq]) * 5;
    
    let total_bonus = corner_bonus + edge_bonus + proximity_bonus;
    
    // Scale by phase and who is winning
    let scaled = (total_bonus * scale) / 128;
    
    if material > 0 {
        S::new(0, scaled as i16) // Only affects endgame
    } else {
        S::new(0, -scaled as i16)
    }
}

/// Quick material balance (positive = white ahead)
#[inline(always)]
fn material_balance(board: &Board) -> i32 {
    const VALUES: [i32; 5] = [100, 320, 330, 500, 900];
    let pieces = [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen];
    
    let mut score = 0i32;
    for (i, &piece) in pieces.iter().enumerate() {
        let white = (board.piece_bb(piece) & board.color_bb(Color::White)).count() as i32;
        let black = (board.piece_bb(piece) & board.color_bb(Color::Black)).count() as i32;
        score += VALUES[i] * (white - black);
    }
    score
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_score() {
        let s = S::new(100, -50);
        assert_eq!(s.mg(), 100);
        assert_eq!(s.eg(), -50);
        
        let s2 = S::new(-200, 150);
        let sum = s + s2;
        assert_eq!(sum.mg(), -100);
        assert_eq!(sum.eg(), 100);
    }

    #[test]
    fn test_starting_position() {
        let board = Board::default();
        let score = evaluate(&board);
        // Should be close to 0 (symmetric position)
        assert!(score.raw().abs() < 50, "Starting position score: {}", score.raw());
    }

    #[test]
    fn test_material_advantage() {
        // White missing queen
        let board = Board::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let score = evaluate(&board);
        // White to move with extra queen should be positive
        assert!(score.raw() > 800, "Queen advantage score: {}", score.raw());
    }

    #[test]
    fn test_phase_calculation() {
        let board = Board::default();
        let phase = calculate_phase(&board);
        // Opening = low phase (close to 0)
        assert!(phase < 50, "Opening phase: {}", phase);
        
        // Endgame position
        let endgame = Board::from_fen("8/8/8/4k3/8/8/4K3/4R3 w - - 0 1").unwrap();
        let eg_phase = calculate_phase(&endgame);
        // Endgame = high phase (close to 256)
        assert!(eg_phase > 200, "Endgame phase: {}", eg_phase);
    }

    #[test]
    fn test_passed_pawn() {
        // White has a passed pawn on d5
        let board = Board::from_fen("8/8/8/3P4/8/8/8/4K2k w - - 0 1").unwrap();
        let score = evaluate(&board);
        // Should have bonus for passed pawn
        assert!(score.raw() > 100, "Passed pawn score: {}", score.raw());
    }
}
