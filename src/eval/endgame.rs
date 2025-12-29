//! Endgame evaluation optimized for finding mates.
//!
//! This evaluation is designed to:
//! - Push the losing king toward corners/edges
//! - Bring the winning king closer to the enemy king
//! - Heavily reward passed pawn advancement
//! - Avoid draw-like positions (3-fold repetition prone)
//!
//! Automatically activates when there are very few pieces on the board.

use crate::types::{Board, Score, Color, Piece, Bitboard};
use movegen::Square;

/// Feature toggle - set to false to disable endgame eval
pub const USE_ENDGAME_EVAL: bool = true;

/// Threshold for switching to endgame eval (non-pawn, non-king pieces)
const ENDGAME_PIECE_THRESHOLD: u32 = 6;

/// Minimum material advantage to use endgame eval (centipawns)
const MATERIAL_ADVANTAGE_THRESHOLD: i32 = 150;

/// Check if we should use endgame evaluation
/// Conditions:
/// - Very few non-pawn/king pieces remain (≤6 total)
/// - OR one side has significant material advantage in a simplified position
pub fn should_use_endgame(board: &Board) -> bool {
    if !USE_ENDGAME_EVAL {
        return false;
    }
    
    let non_pawn_king = board.piece_bb(Piece::Knight) 
        | board.piece_bb(Piece::Bishop) 
        | board.piece_bb(Piece::Rook) 
        | board.piece_bb(Piece::Queen);
    
    let piece_count = non_pawn_king.count();
    
    // Always use in very sparse positions
    if piece_count <= 3 {
        return true;
    }
    
    // Use endgame eval when few pieces remain
    if piece_count <= ENDGAME_PIECE_THRESHOLD {
        // Check for material imbalance - look for winning chances
        let material = material_balance(board);
        if material.abs() >= MATERIAL_ADVANTAGE_THRESHOLD {
            return true;
        }
    }
    
    false
}

/// Distance from square to nearest corner (0-3, lower is more cornered)
#[inline]
fn corner_distance(sq: Square) -> i32 {
    let file = sq.file().index() as i32;
    let rank = sq.rank().index() as i32;
    
    // Distance to nearest edge file (A or H)
    let file_dist = file.min(7 - file);
    // Distance to nearest edge rank (1 or 8)
    let rank_dist = rank.min(7 - rank);
    
    // Manhattan distance to corner is sum of distances to edges
    file_dist + rank_dist
}

/// Distance from square to center (0 = center, higher = edge)
#[inline]
fn center_distance(sq: Square) -> i32 {
    let file = sq.file().index() as i32;
    let rank = sq.rank().index() as i32;
    
    // Distance from center files (d/e = 3/4)
    let file_dist = (file - 3).abs().min((file - 4).abs());
    // Distance from center ranks (4/5)
    let rank_dist = (rank - 3).abs().min((rank - 4).abs());
    
    file_dist + rank_dist
}

/// Chebyshev distance between two squares (king moves to reach)
#[inline]
fn king_distance(sq1: Square, sq2: Square) -> i32 {
    let file1 = sq1.file().index() as i32;
    let rank1 = sq1.rank().index() as i32;
    let file2 = sq2.file().index() as i32;
    let rank2 = sq2.rank().index() as i32;
    
    (file1 - file2).abs().max((rank1 - rank2).abs())
}

/// Calculate material score (positive = white ahead)
fn material_balance(board: &Board) -> i32 {
    let mut score = 0;
    
    const PIECE_VALUES: [(Piece, i32); 5] = [
        (Piece::Pawn, 100),
        (Piece::Knight, 320),
        (Piece::Bishop, 330),
        (Piece::Rook, 500),
        (Piece::Queen, 900),
    ];
    
    for (piece, value) in PIECE_VALUES {
        let white = (board.piece_bb(piece) & board.color_bb(Color::White)).count() as i32;
        let black = (board.piece_bb(piece) & board.color_bb(Color::Black)).count() as i32;
        score += value * (white - black);
    }
    
    score
}

/// Count passed pawns for a color and their advancement
fn passed_pawn_bonus(board: &Board, color: Color) -> i32 {
    let pawns = board.piece_bb(Piece::Pawn) & board.color_bb(color);
    let enemy_pawns = board.piece_bb(Piece::Pawn) & board.color_bb(!color);
    
    let mut bonus = 0;
    
    for sq in pawns {
        let file = sq.file();
        let rank = sq.rank();
        
        // Check if pawn is passed (no enemy pawns blocking or attacking)
        let file_idx = file.index();
        let rank_idx = rank.index();
        
        // Build mask for blocking squares
        let mut blocking_mask = 0u64;
        let files_to_check = [
            file_idx.saturating_sub(1),
            file_idx,
            (file_idx + 1).min(7),
        ];
        
        for &f in &files_to_check {
            if color == Color::White {
                // Check ranks above for white
                for r in (rank_idx + 1)..8 {
                    blocking_mask |= 1u64 << (r * 8 + f);
                }
            } else {
                // Check ranks below for black
                for r in 0..rank_idx {
                    blocking_mask |= 1u64 << (r * 8 + f);
                }
            }
        }
        
        let blocking = Bitboard::new(blocking_mask);
        
        if (enemy_pawns & blocking).is_empty() {
            // It's a passed pawn! Bonus based on advancement
            let advancement = if color == Color::White {
                rank_idx as i32
            } else {
                7 - rank_idx as i32
            };
            
            // Exponential bonus for advanced passed pawns
            bonus += match advancement {
                6 => 200, // About to promote!
                5 => 120,
                4 => 60,
                3 => 30,
                2 => 15,
                _ => 5,
            };
        }
    }
    
    bonus
}

/// Endgame evaluation from white's perspective
/// Optimized to find mates by:
/// 1. Pushing enemy king to corners/edges
/// 2. Bringing our king closer
/// 3. Rewarding passed pawn advancement
pub fn evaluate(board: &Board) -> Score {
    let material = material_balance(board);
    
    // Determine winning side
    let winning_side = if material > 50 {
        Color::White
    } else if material < -50 {
        Color::Black
    } else {
        // Material is roughly equal, use HCE instead
        return crate::eval::hce::evaluate(board);
    };
    
    let losing_side = !winning_side;
    
    let winner_king = board.king_square(winning_side);
    let loser_king = board.king_square(losing_side);
    
    // Base score is material (scaled up for winning endgames)
    let material_mult = if material.abs() > 300 { 2 } else { 1 };
    let mut score = material * material_mult;
    
    // === KING CORNER DRIVING ===
    // Reward pushing enemy king toward corners HEAVILY
    // corner_distance: 0 = corner, 6 = center
    let loser_corner_dist = corner_distance(loser_king);
    let corner_bonus = (6 - loser_corner_dist) * 40; // Up to 240cp bonus
    
    // Extra bonus for actually being in corner (distance 0)
    let corner_trap = if loser_corner_dist == 0 { 100 } else { 0 };
    
    // === KING PROXIMITY ===
    // Reward bringing our king closer to enemy king
    // king_distance: 1-7 squares
    let king_dist = king_distance(winner_king, loser_king);
    let proximity_bonus = (7 - king_dist) * 25; // Up to 150cp bonus
    
    // === EDGE DRIVING ===
    // Also reward pushing enemy king to edges (not just corners)
    let loser_center_dist = center_distance(loser_king);
    let edge_bonus = loser_center_dist * 20; // Up to ~80cp bonus
    
    // Extra bonus for being on edge (center_dist >= 3)
    let edge_trap = if loser_center_dist >= 3 { 50 } else { 0 };
    
    // === WINNING KING CENTRALIZATION ===
    // Our king should be somewhat centralized to control squares
    let winner_center_dist = center_distance(winner_king);
    let centralization = (4 - winner_center_dist) * 10; // Up to 40cp bonus
    
    // === PASSED PAWN ADVANCEMENT ===
    let white_passed = passed_pawn_bonus(board, Color::White);
    let black_passed = passed_pawn_bonus(board, Color::Black);
    let passed_diff = white_passed - black_passed;
    
    // Apply bonuses based on winning side
    let endgame_bonus = if winning_side == Color::White {
        corner_bonus + corner_trap + proximity_bonus + edge_bonus + edge_trap + centralization
    } else {
        -(corner_bonus + corner_trap + proximity_bonus + edge_bonus + edge_trap + centralization)
    };
    
    score += endgame_bonus;
    score += passed_diff;
    
    // === ANTI-DRAW BIAS ===
    // Add a small random-ish component based on position to avoid repetition
    // This makes the engine prefer variety in drawn-ish positions
    let hash_component = (board.hash() % 11) as i32 - 5; // -5 to +5
    if material.abs() > 100 {
        score += hash_component;
    }
    
    // Return from side-to-move perspective
    if board.turn() == Color::White {
        Score::cp(score)
    } else {
        Score::cp(-score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_corner_distance() {
        assert_eq!(corner_distance(Square::A1), 0); // Corner
        assert_eq!(corner_distance(Square::H8), 0); // Corner
        assert_eq!(corner_distance(Square::A8), 0); // Corner
        assert_eq!(corner_distance(Square::E4), 6); // Near center
        assert_eq!(corner_distance(Square::D4), 6); // Center
    }
    
    #[test]
    fn test_king_distance() {
        assert_eq!(king_distance(Square::E1, Square::E8), 7);
        assert_eq!(king_distance(Square::A1, Square::B2), 1);
        assert_eq!(king_distance(Square::A1, Square::H8), 7);
    }
    
    #[test]
    fn test_krk_endgame() {
        // KRK endgame - white should win
        let board = Board::from_fen("8/8/8/4k3/8/8/4K3/4R3 w - - 0 1").unwrap();
        assert!(should_use_endgame(&board));
        
        let score = evaluate(&board);
        // White has rook advantage, should be positive
        assert!(score.raw() > 400);
    }
    
    #[test]
    fn test_king_in_corner_bonus() {
        // Enemy king in corner should give bonus
        let board1 = Board::from_fen("k7/8/8/8/8/8/4K3/4R3 w - - 0 1").unwrap();
        let board2 = Board::from_fen("4k3/8/8/8/8/8/4K3/4R3 w - - 0 1").unwrap();
        
        let score1 = evaluate(&board1); // King in corner
        let score2 = evaluate(&board2); // King in center
        
        // Corner position should score higher
        assert!(score1.raw() > score2.raw());
    }
}
