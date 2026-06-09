//! Pawn move generation.

use super::{Move, MoveFlag, MoveSink};
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::types::{Color, Piece, Square, Rank};
use crate::attacks::{pawn_attacks, line};

impl Board {
    /// Generate all pawn moves.
    pub fn generate_pawn_moves<M: MoveSink>(&self, moves: &mut M, target_mask: Bitboard, pinned: Bitboard) {
        let us = self.turn();
        let pawns = self.piece_color_bb(Piece::Pawn, us);
        let empty = self.empty_squares();
        let enemies = self.them();
        let king_sq = self.king_square(us);
        
        let (push_dir, start_rank, promo_rank): (i8, Rank, Rank) = if us == Color::White {
            (8, Rank::R2, Rank::R7)
        } else {
            (-8, Rank::R7, Rank::R2)
        };
        
        // Single pushes
        let single_push_targets = if us == Color::White {
            (pawns & !Bitboard::rank_mask(promo_rank)).north() & empty
        } else {
            (pawns & !Bitboard::rank_mask(promo_rank)).south() & empty
        };
        
        for to in single_push_targets & target_mask {
            let from = unsafe { Square::from_index_unchecked((to.index() as i8 - push_dir) as u8) };
            
            // Check pin constraint
            if pinned.contains(from) {
                if !line(king_sq, from).contains(to) {
                    continue;
                }
            }
            
            moves.push(Move::new(from, to, MoveFlag::Quiet));
        }
        
        // Double pushes
        let double_push_pawns = pawns & Bitboard::rank_mask(start_rank);
        let single_step = if us == Color::White {
            double_push_pawns.north() & empty
        } else {
            double_push_pawns.south() & empty
        };
        let double_push_targets = if us == Color::White {
            single_step.north() & empty & Bitboard::RANK_4
        } else {
            single_step.south() & empty & Bitboard::RANK_5
        };
        
        for to in double_push_targets & target_mask {
            let from = unsafe { Square::from_index_unchecked((to.index() as i8 - 2 * push_dir) as u8) };
            
            if pinned.contains(from) {
                if !line(king_sq, from).contains(to) {
                    continue;
                }
            }
            
            moves.push(Move::new(from, to, MoveFlag::DoublePawnPush));
        }
        
        // Captures
        for from in pawns & !Bitboard::rank_mask(promo_rank) {
            let attacks = pawn_attacks(us, from) & enemies & target_mask;
            
            for to in attacks {
                if pinned.contains(from) {
                    if !line(king_sq, from).contains(to) {
                        continue;
                    }
                }
                
                moves.push(Move::new(from, to, MoveFlag::Capture));
            }
        }
        
        // Promotions
        let promo_pawns = pawns & Bitboard::rank_mask(promo_rank);
        
        for from in promo_pawns {
            // Push promotion
            let to = if us == Color::White {
                from.north()
            } else {
                from.south()
            };
            
            if let Some(to) = to {
                if empty.contains(to) && target_mask.contains(to) {
                    if !pinned.contains(from) || line(king_sq, from).contains(to) {
                        self.add_promotions(moves, from, to, false);
                    }
                }
            }
            
            // Capture promotions
            let attacks = pawn_attacks(us, from) & enemies & target_mask;
            for to in attacks {
                if !pinned.contains(from) || line(king_sq, from).contains(to) {
                    self.add_promotions(moves, from, to, true);
                }
            }
        }
        
        // En passant
        if let Some(ep_sq) = self.ep_square() {
            self.generate_en_passant(moves, ep_sq, pinned);
        }
    }

    /// Add all four promotion moves.
    fn add_promotions<M: MoveSink>(&self, moves: &mut M, from: Square, to: Square, capture: bool) {
        moves.push(Move::new(from, to, MoveFlag::promotion(Piece::Queen, capture)));
        moves.push(Move::new(from, to, MoveFlag::promotion(Piece::Rook, capture)));
        moves.push(Move::new(from, to, MoveFlag::promotion(Piece::Bishop, capture)));
        moves.push(Move::new(from, to, MoveFlag::promotion(Piece::Knight, capture)));
    }

    /// Generate en passant moves with special legality check.
    fn generate_en_passant<M: MoveSink>(&self, moves: &mut M, ep_sq: Square, _pinned: Bitboard) {
        let us = self.turn();
        let pawns = self.piece_color_bb(Piece::Pawn, us);
        
        // Potential capturing pawns
        let attackers = pawn_attacks(!us, ep_sq) & pawns;
        
        for from in attackers {
            // En passant has special pin/discovery rules
            // Need to check if capturing pawn or captured pawn was blocking check
            
            let cap_sq = if us == Color::White {
                unsafe { Square::from_index_unchecked(ep_sq.index() - 8) }
            } else {
                unsafe { Square::from_index_unchecked(ep_sq.index() + 8) }
            };
            
            // Simulate the move
            let king_sq = self.king_square(us);
            let occ = (self.occupied() ^ Bitboard::from_square(from) ^ Bitboard::from_square(cap_sq))
                | Bitboard::from_square(ep_sq);
            
            // Check if king is attacked after EP
            let rooks = (self.piece_bb(Piece::Rook) | self.piece_bb(Piece::Queen)) & self.them();
            let bishops = (self.piece_bb(Piece::Bishop) | self.piece_bb(Piece::Queen)) & self.them();
            
            let rook_attacks = crate::attacks::rook_attacks(king_sq, occ);
            let bishop_attacks = crate::attacks::bishop_attacks(king_sq, occ);
            
            if (rook_attacks & rooks).any() || (bishop_attacks & bishops).any() {
                continue; // EP would leave king in check
            }
            
            moves.push(Move::new(from, ep_sq, MoveFlag::EnPassant));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MoveList;

    #[test]
    fn test_pawn_pushes() {
        let board = Board::startpos();
        let mut moves = MoveList::new();
        board.generate_pawn_moves(&mut moves, Bitboard::UNIVERSE, Bitboard::EMPTY);
        
        // 8 single pushes + 8 double pushes = 16
        assert_eq!(moves.len(), 16);
    }

    #[test]
    fn test_pawn_captures() {
        let board = Board::from_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
        let mut moves = MoveList::new();
        board.generate_pawn_moves(&mut moves, Bitboard::UNIVERSE, Bitboard::EMPTY);
        
        // 1 push + 1 capture = 2
        assert_eq!(moves.len(), 2);
    }
}
