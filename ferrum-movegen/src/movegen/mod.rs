//! Move generation logic.

mod moves;
mod pawns;
mod knights;
mod king;
mod sliders;
mod legality;

pub use moves::{Move, MoveFlag, MoveList, ScoredMove, MoveSink, MoveCounter};

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::types::{Color, Piece, Square};
use crate::attacks::{knight_attacks, king_attacks, bishop_attacks, rook_attacks, between};

impl Board {
    /// Generate all legal moves.
    pub fn generate_moves(&self) -> MoveList {
        let mut moves = MoveList::new();
        self.generate_moves_impl(&mut moves);
        moves
    }

    /// Generate count of all legal moves (bulk count).
    pub fn generate_moves_count(&self) -> u64 {
        let mut counter = MoveCounter::new();
        self.generate_moves_impl(&mut counter);
        counter.count
    }

    /// Generic move generation implementation.
    fn generate_moves_impl<M: MoveSink>(&self, moves: &mut M) {
        if self.checkers().more_than_one() {
            // Double check: only king moves are legal
            self.generate_king_moves(moves);
        } else if self.checkers().any() {
            // Single check: generate evasions
            self.generate_evasions(moves);
        } else {
            // Not in check: generate all moves
            self.generate_all_moves(moves);
        }
    }

    /// Generate all pseudolegal moves (no check test).
    fn generate_all_moves<M: MoveSink>(&self, moves: &mut M) {
        let pinned = self.compute_pinned();
        let target = !self.us(); // Can move to empty or enemy squares
        
        self.generate_pawn_moves(moves, Bitboard::UNIVERSE, pinned);
        self.generate_knight_moves(moves, target, pinned);
        self.generate_bishop_moves(moves, target, pinned);
        self.generate_rook_moves(moves, target, pinned);
        self.generate_queen_moves(moves, target, pinned);
        self.generate_king_moves(moves);
    }

    /// Generate only capture moves (staged generation).
    pub fn generate_captures(&self) -> MoveList {
        let mut moves = MoveList::new();
        if self.checkers().more_than_one() {
            self.generate_king_moves(&mut moves);
        } else if self.checkers().any() {
            self.generate_evasions(&mut moves);
        } else {
            self.generate_captures_impl(&mut moves);
        }
        // Filter to only captures
        let mut result = MoveList::new();
        for m in moves.iter() {
            if m.is_capture() {
                result.push(m);
            }
        }
        result
    }

    /// Generate only quiet moves (non-captures, staged generation).
    pub fn generate_quiets(&self) -> MoveList {
        let mut moves = MoveList::new();
        if self.checkers().more_than_one() {
            self.generate_king_moves(&mut moves);
        } else if self.checkers().any() {
            self.generate_evasions(&mut moves);
        } else {
            self.generate_quiets_impl(&mut moves);
        }
        // Filter to only quiets
        let mut result = MoveList::new();
        for m in moves.iter() {
            if !m.is_capture() {
                result.push(m);
            }
        }
        result
    }

    /// Internal: generate captures only.
    fn generate_captures_impl<M: MoveSink>(&self, moves: &mut M) {
        let pinned = self.compute_pinned();
        let target = self.them(); // Only enemy squares
        
        self.generate_pawn_moves(moves, target, pinned);
        self.generate_knight_moves(moves, target, pinned);
        self.generate_bishop_moves(moves, target, pinned);
        self.generate_rook_moves(moves, target, pinned);
        self.generate_queen_moves(moves, target, pinned);
        self.generate_king_moves(moves);
    }

    /// Internal: generate quiets only.
    fn generate_quiets_impl<M: MoveSink>(&self, moves: &mut M) {
        let pinned = self.compute_pinned();
        let target = !self.occupied(); // Only empty squares
        
        self.generate_pawn_moves(moves, target, pinned);
        self.generate_knight_moves(moves, target, pinned);
        self.generate_bishop_moves(moves, target, pinned);
        self.generate_rook_moves(moves, target, pinned);
        self.generate_queen_moves(moves, target, pinned);
        self.generate_king_moves(moves);
    }

    /// Generate moves when in check.
    fn generate_evasions<M: MoveSink>(&self, moves: &mut M) {
        let king_sq = self.king_square(self.turn());
        let checker_sq = unsafe { self.checkers().lsb_unchecked() };
        
        // Squares that block or capture the checker
        let block_mask = between(king_sq, checker_sq) | self.checkers();
        let pinned = self.compute_pinned();
        
        // Pawn moves that block/capture
        self.generate_pawn_moves(moves, block_mask, pinned);
        
        // Knight moves that block/capture
        self.generate_knight_moves(moves, block_mask & !self.us(), pinned);
        
        // Slider moves that block/capture
        self.generate_bishop_moves(moves, block_mask & !self.us(), pinned);
        self.generate_rook_moves(moves, block_mask & !self.us(), pinned);
        self.generate_queen_moves(moves, block_mask & !self.us(), pinned);
        
        // King moves (always generated)
        self.generate_king_moves(moves);
    }

    /// Compute pinned pieces.
    pub fn compute_pinned(&self) -> Bitboard {
        let king_sq = self.king_square(self.turn());
        let occ = self.occupied();
        let us = self.us();
        let them = self.them();
        
        let mut pinned = Bitboard::EMPTY;
        
        // Diagonal pinners
        let diag_sliders = (self.piece_bb(Piece::Bishop) | self.piece_bb(Piece::Queen)) & them;
        let potential_diag = bishop_attacks(king_sq, Bitboard::EMPTY) & diag_sliders;
        
        for pinner in potential_diag {
            let between_bb = between(king_sq, pinner);
            let blockers = between_bb & occ;
            if blockers.exactly_one() && (blockers & us).any() {
                pinned |= blockers;
            }
        }
        
        // Orthogonal pinners
        let ortho_sliders = (self.piece_bb(Piece::Rook) | self.piece_bb(Piece::Queen)) & them;
        let potential_ortho = rook_attacks(king_sq, Bitboard::EMPTY) & ortho_sliders;
        
        for pinner in potential_ortho {
            let between_bb = between(king_sq, pinner);
            let blockers = between_bb & occ;
            if blockers.exactly_one() && (blockers & us).any() {
                pinned |= blockers;
            }
        }
        
        pinned
    }

    /// Generate knight moves.
    fn generate_knight_moves<M: MoveSink>(&self, moves: &mut M, target: Bitboard, pinned: Bitboard) {
        let knights = self.piece_color_bb(Piece::Knight, self.turn()) & !pinned;
        
        for from in knights {
            let attacks = knight_attacks(from) & target;
            for to in attacks {
                let flag = if self.them().contains(to) {
                    MoveFlag::Capture
                } else {
                    MoveFlag::Quiet
                };
                moves.push(Move::new(from, to, flag));
            }
        }
    }

    /// Generate bishop moves.
    fn generate_bishop_moves<M: MoveSink>(&self, moves: &mut M, target: Bitboard, pinned: Bitboard) {
        let bishops = self.piece_color_bb(Piece::Bishop, self.turn());
        let occ = self.occupied();
        let king_sq = self.king_square(self.turn());
        
        for from in bishops {
            let mut attacks = bishop_attacks(from, occ) & target;
            
            // If pinned, can only move along pin ray
            if pinned.contains(from) {
                attacks &= crate::attacks::line(king_sq, from);
            }
            
            for to in attacks {
                let flag = if self.them().contains(to) {
                    MoveFlag::Capture
                } else {
                    MoveFlag::Quiet
                };
                moves.push(Move::new(from, to, flag));
            }
        }
    }

    /// Generate rook moves.
    fn generate_rook_moves<M: MoveSink>(&self, moves: &mut M, target: Bitboard, pinned: Bitboard) {
        let rooks = self.piece_color_bb(Piece::Rook, self.turn());
        let occ = self.occupied();
        let king_sq = self.king_square(self.turn());
        
        for from in rooks {
            let mut attacks = rook_attacks(from, occ) & target;
            
            if pinned.contains(from) {
                attacks &= crate::attacks::line(king_sq, from);
            }
            
            for to in attacks {
                let flag = if self.them().contains(to) {
                    MoveFlag::Capture
                } else {
                    MoveFlag::Quiet
                };
                moves.push(Move::new(from, to, flag));
            }
        }
    }

    /// Generate queen moves.
    fn generate_queen_moves<M: MoveSink>(&self, moves: &mut M, target: Bitboard, pinned: Bitboard) {
        let queens = self.piece_color_bb(Piece::Queen, self.turn());
        let occ = self.occupied();
        let king_sq = self.king_square(self.turn());
        
        for from in queens {
            let mut attacks = (bishop_attacks(from, occ) | rook_attacks(from, occ)) & target;
            
            if pinned.contains(from) {
                attacks &= crate::attacks::line(king_sq, from);
            }
            
            for to in attacks {
                let flag = if self.them().contains(to) {
                    MoveFlag::Capture
                } else {
                    MoveFlag::Quiet
                };
                moves.push(Move::new(from, to, flag));
            }
        }
    }

    /// Generate king moves including castling.
    fn generate_king_moves<M: MoveSink>(&self, moves: &mut M) {
        let king_sq = self.king_square(self.turn());
        let occ = self.occupied();
        
        // Normal king moves
        let attacks = king_attacks(king_sq) & !self.us();
        
        for to in attacks {
            // Check if target square is attacked by enemy
            let new_occ = (occ ^ Bitboard::from_square(king_sq)) | Bitboard::from_square(to);
            if (self.attackers_to(to, new_occ) & self.them()).is_empty() {
                let flag = if self.them().contains(to) {
                    MoveFlag::Capture
                } else {
                    MoveFlag::Quiet
                };
                moves.push(Move::new(king_sq, to, flag));
            }
        }
        
        // Castling (only if not in check)
        if self.checkers().is_empty() {
            self.generate_castling_moves(moves, king_sq);
        }
    }

    /// Generate castling moves.
    fn generate_castling_moves<M: MoveSink>(&self, moves: &mut M, king_sq: Square) {
        let us = self.turn();
        let occ = self.occupied();
        
        if us == Color::White {
            // Kingside
            if self.castling().has_white_kingside() {
                if (occ & Bitboard::BETWEEN_E1_G1).is_empty() {
                    // Check intermediate squares not attacked
                    if (self.attackers_to(Square::F1, occ) & self.them()).is_empty()
                        && (self.attackers_to(Square::G1, occ) & self.them()).is_empty()
                    {
                        moves.push(Move::new(king_sq, Square::G1, MoveFlag::KingCastle));
                    }
                }
            }
            // Queenside
            if self.castling().has_white_queenside() {
                if (occ & Bitboard::BETWEEN_E1_C1).is_empty() {
                    if (self.attackers_to(Square::D1, occ) & self.them()).is_empty()
                        && (self.attackers_to(Square::C1, occ) & self.them()).is_empty()
                    {
                        moves.push(Move::new(king_sq, Square::C1, MoveFlag::QueenCastle));
                    }
                }
            }
        } else {
            // Kingside
            if self.castling().has_black_kingside() {
                if (occ & Bitboard::BETWEEN_E8_G8).is_empty() {
                    if (self.attackers_to(Square::F8, occ) & self.them()).is_empty()
                        && (self.attackers_to(Square::G8, occ) & self.them()).is_empty()
                    {
                        moves.push(Move::new(king_sq, Square::G8, MoveFlag::KingCastle));
                    }
                }
            }
            // Queenside
            if self.castling().has_black_queenside() {
                if (occ & Bitboard::BETWEEN_E8_C8).is_empty() {
                    if (self.attackers_to(Square::D8, occ) & self.them()).is_empty()
                        && (self.attackers_to(Square::C8, occ) & self.them()).is_empty()
                    {
                        moves.push(Move::new(king_sq, Square::C8, MoveFlag::QueenCastle));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_startpos_moves() {
        let board = Board::startpos();
        let moves = board.generate_moves();
        // 20 legal moves in starting position
        assert_eq!(moves.len(), 20);
    }

    #[test]
    fn test_kiwipete_moves() {
        let board = Board::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        ).unwrap();
        let moves = board.generate_moves();
        // Kiwipete has 48 legal moves
        assert_eq!(moves.len(), 48);
    }
}
