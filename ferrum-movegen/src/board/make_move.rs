//! Make and unmake move logic.

use super::Board;
use super::zobrist::ZOBRIST;
use crate::bitboard::Bitboard;
use crate::types::{Square, Piece, Color, CastleRights};
use crate::movegen::{Move, MoveFlag};

/// State that needs to be saved for unmaking a move.
#[derive(Clone, Copy)]
pub struct UndoInfo {
    pub castling: CastleRights,
    pub ep_square: Option<Square>,
    pub halfmove_clock: u8,
    pub hash: u64,
    pub checkers: Bitboard,
    pub captured: Option<Piece>,
}

impl Board {
    /// Make a move and return a new board (copy-make pattern).
    /// This is the recommended method for search.
    #[inline]
    pub fn make_move_new(&self, mv: Move) -> Board {
        let mut result = *self;
        
        let from = mv.from();
        let to = mv.to();
        let flag = mv.flag();
        let us = self.turn;
        let them = !us;

        // Find the moving piece
        let piece = self.piece_at(from).map(|(p, _)| p).unwrap_or(Piece::Pawn);

        // === Zobrist hash updates ===
        
        // XOR out old en passant
        if let Some(ep) = self.ep_square {
            result.hash ^= ZOBRIST.ep_file(ep.file());
        }
        result.ep_square = None;
        
        // XOR out old castling rights
        result.hash ^= ZOBRIST.castling(self.castling);

        match flag {
            MoveFlag::Quiet => {
                // Move piece: XOR out from, XOR in to
                result.hash ^= ZOBRIST.piece_square(piece, us, from);
                result.hash ^= ZOBRIST.piece_square(piece, us, to);
                result.move_piece_fast(from, to, piece, us);
            }
            MoveFlag::DoublePawnPush => {
                result.hash ^= ZOBRIST.piece_square(piece, us, from);
                result.hash ^= ZOBRIST.piece_square(piece, us, to);
                result.move_piece_fast(from, to, piece, us);
                let ep = if us == Color::White {
                    unsafe { Square::from_index_unchecked(to.index() - 8) }
                } else {
                    unsafe { Square::from_index_unchecked(to.index() + 8) }
                };
                result.ep_square = Some(ep);
                result.hash ^= ZOBRIST.ep_file(ep.file());
            }
            MoveFlag::Capture => {
                // Remove captured piece
                if let Some((cap_piece, _)) = self.piece_at(to) {
                    result.hash ^= ZOBRIST.piece_square(cap_piece, them, to);
                    result.remove_piece_fast(to, cap_piece, them);
                }
                // Move capturing piece
                result.hash ^= ZOBRIST.piece_square(piece, us, from);
                result.hash ^= ZOBRIST.piece_square(piece, us, to);
                result.move_piece_fast(from, to, piece, us);
            }
            MoveFlag::EnPassant => {
                let cap_sq = if us == Color::White {
                    unsafe { Square::from_index_unchecked(to.index() - 8) }
                } else {
                    unsafe { Square::from_index_unchecked(to.index() + 8) }
                };
                // Remove captured pawn
                result.hash ^= ZOBRIST.piece_square(Piece::Pawn, them, cap_sq);
                result.remove_piece_fast(cap_sq, Piece::Pawn, them);
                // Move pawn
                result.hash ^= ZOBRIST.piece_square(Piece::Pawn, us, from);
                result.hash ^= ZOBRIST.piece_square(Piece::Pawn, us, to);
                result.move_piece_fast(from, to, Piece::Pawn, us);
            }
            MoveFlag::KingCastle => {
                // Move king
                result.hash ^= ZOBRIST.piece_square(Piece::King, us, from);
                result.hash ^= ZOBRIST.piece_square(Piece::King, us, to);
                result.move_piece_fast(from, to, Piece::King, us);
                // Move rook
                let (rook_from, rook_to) = if us == Color::White {
                    (Square::H1, Square::F1)
                } else {
                    (Square::H8, Square::F8)
                };
                result.hash ^= ZOBRIST.piece_square(Piece::Rook, us, rook_from);
                result.hash ^= ZOBRIST.piece_square(Piece::Rook, us, rook_to);
                result.move_piece_fast(rook_from, rook_to, Piece::Rook, us);
            }
            MoveFlag::QueenCastle => {
                // Move king
                result.hash ^= ZOBRIST.piece_square(Piece::King, us, from);
                result.hash ^= ZOBRIST.piece_square(Piece::King, us, to);
                result.move_piece_fast(from, to, Piece::King, us);
                // Move rook
                let (rook_from, rook_to) = if us == Color::White {
                    (Square::A1, Square::D1)
                } else {
                    (Square::A8, Square::D8)
                };
                result.hash ^= ZOBRIST.piece_square(Piece::Rook, us, rook_from);
                result.hash ^= ZOBRIST.piece_square(Piece::Rook, us, rook_to);
                result.move_piece_fast(rook_from, rook_to, Piece::Rook, us);
            }
            _ if flag.is_promotion() => {
                let promo_piece = flag.promotion_piece().unwrap();
                // Remove pawn
                result.hash ^= ZOBRIST.piece_square(Piece::Pawn, us, from);
                result.remove_piece_fast(from, Piece::Pawn, us);
                // If capture, remove captured piece
                if flag.is_capture() {
                    if let Some((cap_piece, _)) = self.piece_at(to) {
                        result.hash ^= ZOBRIST.piece_square(cap_piece, them, to);
                        result.remove_piece_fast(to, cap_piece, them);
                    }
                }
                // Add promoted piece
                result.hash ^= ZOBRIST.piece_square(promo_piece, us, to);
                result.add_piece_fast(to, promo_piece, us);
            }
            _ => {}
        }

        // Update castling rights
        let new_castling = result.castling.remove(CastleRights::update_mask(from))
                                          .remove(CastleRights::update_mask(to));
        result.castling = new_castling;
        result.hash ^= ZOBRIST.castling(new_castling);

        // Switch side
        result.turn = them;
        result.hash ^= ZOBRIST.side();

        // Update checkers
        result.update_checkers();

        result
    }
    /// Make a move on the board.
    /// Returns undo information for unmaking the move.
    pub fn make_move(&mut self, mv: Move) -> UndoInfo {
        let undo = UndoInfo {
            castling: self.castling,
            ep_square: self.ep_square,
            halfmove_clock: self.halfmove_clock,
            hash: self.hash,
            checkers: self.checkers,
            captured: None,
        };

        let from = mv.from();
        let to = mv.to();
        let flag = mv.flag();
        let us = self.turn;
        let them = !us;

        // Find the moving piece
        let piece = self.piece_at(from).map(|(p, _)| p).unwrap_or(Piece::Pawn);

        // Handle en passant hash
        if let Some(ep) = self.ep_square {
            self.hash ^= ZOBRIST.ep_file(ep.file());
        }
        self.ep_square = None;

        // Handle castling rights hash
        self.hash ^= ZOBRIST.castling(self.castling);

        // Track captured piece
        let mut captured: Option<Piece> = None;

        match flag {
            MoveFlag::Quiet | MoveFlag::DoublePawnPush => {
                self.move_piece(from, to, piece, us);
                
                if flag == MoveFlag::DoublePawnPush {
                    // Set en passant square
                    let ep = if us == Color::White {
                        unsafe { Square::from_index_unchecked(to.index() - 8) }
                    } else {
                        unsafe { Square::from_index_unchecked(to.index() + 8) }
                    };
                    self.ep_square = Some(ep);
                    self.hash ^= ZOBRIST.ep_file(ep.file());
                }
            }
            MoveFlag::Capture => {
                // Remove captured piece
                if let Some((cap_piece, _)) = self.piece_at(to) {
                    captured = Some(cap_piece);
                    self.remove_piece(to, cap_piece, them);
                }
                self.move_piece(from, to, piece, us);
            }
            MoveFlag::EnPassant => {
                let cap_sq = if us == Color::White {
                    unsafe { Square::from_index_unchecked(to.index() - 8) }
                } else {
                    unsafe { Square::from_index_unchecked(to.index() + 8) }
                };
                captured = Some(Piece::Pawn);
                self.remove_piece(cap_sq, Piece::Pawn, them);
                self.move_piece(from, to, Piece::Pawn, us);
            }
            MoveFlag::KingCastle => {
                // Move king
                self.move_piece(from, to, Piece::King, us);
                // Move rook
                let (rook_from, rook_to) = if us == Color::White {
                    (Square::H1, Square::F1)
                } else {
                    (Square::H8, Square::F8)
                };
                self.move_piece(rook_from, rook_to, Piece::Rook, us);
            }
            MoveFlag::QueenCastle => {
                // Move king
                self.move_piece(from, to, Piece::King, us);
                // Move rook
                let (rook_from, rook_to) = if us == Color::White {
                    (Square::A1, Square::D1)
                } else {
                    (Square::A8, Square::D8)
                };
                self.move_piece(rook_from, rook_to, Piece::Rook, us);
            }
            _ if flag.is_promotion() => {
                let promo_piece = flag.promotion_piece().unwrap();
                
                // Remove pawn
                self.remove_piece(from, Piece::Pawn, us);
                
                // Capture if applicable
                if flag.is_capture() {
                    if let Some((cap_piece, _)) = self.piece_at(to) {
                        captured = Some(cap_piece);
                        self.remove_piece(to, cap_piece, them);
                    }
                }
                
                // Add promoted piece
                self.add_piece(to, promo_piece, us);
            }
            _ => {}
        }

        // Update castling rights
        self.castling = self.castling.remove(CastleRights::update_mask(from));
        self.castling = self.castling.remove(CastleRights::update_mask(to));
        self.hash ^= ZOBRIST.castling(self.castling);

        // Update halfmove clock
        if piece == Piece::Pawn || captured.is_some() {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }

        // Update fullmove number
        if us == Color::Black {
            self.fullmove_number += 1;
        }

        // Switch side
        self.turn = them;
        self.hash ^= ZOBRIST.side();

        // Update checkers (can be skipped with make_move_fast for perft)
        self.update_checkers();

        UndoInfo { captured, ..undo }
    }

    /// Make a move without updating checkers (faster for perft).
    /// The checkers bitboard will be invalid after this call.
    #[inline(always)]
    pub fn make_move_fast(&mut self, mv: &Move) -> UndoInfo {
        let undo = UndoInfo {
            castling: self.castling,
            ep_square: self.ep_square,
            halfmove_clock: self.halfmove_clock,
            hash: self.hash,
            checkers: self.checkers,
            captured: None,
        };

        let from = mv.from();
        let to = mv.to();
        let flag = mv.flag();
        let us = self.turn;
        let them = !us;

        // Find the moving piece
        let piece = self.piece_at(from).map(|(p, _)| p).unwrap_or(Piece::Pawn);

        // Clear en passant
        self.ep_square = None;

        // Track captured piece
        let mut captured: Option<Piece> = None;

        match flag {
            MoveFlag::Quiet => {
                self.move_piece_fast(from, to, piece, us);
            }
            MoveFlag::DoublePawnPush => {
                self.move_piece_fast(from, to, piece, us);
                let ep = if us == Color::White {
                    unsafe { Square::from_index_unchecked(to.index() - 8) }
                } else {
                    unsafe { Square::from_index_unchecked(to.index() + 8) }
                };
                self.ep_square = Some(ep);
            }
            MoveFlag::Capture => {
                if let Some((cap_piece, _)) = self.piece_at(to) {
                    captured = Some(cap_piece);
                    self.remove_piece_fast(to, cap_piece, them);
                }
                self.move_piece_fast(from, to, piece, us);
            }
            MoveFlag::EnPassant => {
                let cap_sq = if us == Color::White {
                    unsafe { Square::from_index_unchecked(to.index() - 8) }
                } else {
                    unsafe { Square::from_index_unchecked(to.index() + 8) }
                };
                captured = Some(Piece::Pawn);
                self.remove_piece_fast(cap_sq, Piece::Pawn, them);
                self.move_piece_fast(from, to, Piece::Pawn, us);
            }
            MoveFlag::KingCastle => {
                self.move_piece_fast(from, to, Piece::King, us);
                let (rook_from, rook_to) = if us == Color::White {
                    (Square::H1, Square::F1)
                } else {
                    (Square::H8, Square::F8)
                };
                self.move_piece_fast(rook_from, rook_to, Piece::Rook, us);
            }
            MoveFlag::QueenCastle => {
                self.move_piece_fast(from, to, Piece::King, us);
                let (rook_from, rook_to) = if us == Color::White {
                    (Square::A1, Square::D1)
                } else {
                    (Square::A8, Square::D8)
                };
                self.move_piece_fast(rook_from, rook_to, Piece::Rook, us);
            }
            _ if flag.is_promotion() => {
                let promo_piece = flag.promotion_piece().unwrap();
                self.remove_piece_fast(from, Piece::Pawn, us);
                if flag.is_capture() {
                    if let Some((cap_piece, _)) = self.piece_at(to) {
                        captured = Some(cap_piece);
                        self.remove_piece_fast(to, cap_piece, them);
                    }
                }
                self.add_piece_fast(to, promo_piece, us);
            }
            _ => {}
        }

        // Update castling rights
        self.castling = self.castling.remove(CastleRights::update_mask(from));
        self.castling = self.castling.remove(CastleRights::update_mask(to));

        // Switch side
        self.turn = them;

        UndoInfo { captured, ..undo }
    }

    /// Unmake a move on the board.
    pub fn unmake_move(&mut self, mv: Move, undo: UndoInfo) {
        let from = mv.from();
        let to = mv.to();
        let flag = mv.flag();
        
        // Switch side back
        self.turn = !self.turn;
        let us = self.turn;
        let them = !us;

        // Find the moving piece (now at 'to')
        let piece = if flag.is_promotion() {
            Piece::Pawn
        } else {
            self.piece_at(to).map(|(p, _)| p).unwrap_or(Piece::Pawn)
        };

        match flag {
            MoveFlag::Quiet | MoveFlag::DoublePawnPush => {
                self.move_piece(to, from, piece, us);
            }
            MoveFlag::Capture => {
                self.move_piece(to, from, piece, us);
                if let Some(cap_piece) = undo.captured {
                    self.add_piece(to, cap_piece, them);
                }
            }
            MoveFlag::EnPassant => {
                self.move_piece(to, from, Piece::Pawn, us);
                let cap_sq = if us == Color::White {
                    unsafe { Square::from_index_unchecked(to.index() - 8) }
                } else {
                    unsafe { Square::from_index_unchecked(to.index() + 8) }
                };
                self.add_piece(cap_sq, Piece::Pawn, them);
            }
            MoveFlag::KingCastle => {
                self.move_piece(to, from, Piece::King, us);
                let (rook_from, rook_to) = if us == Color::White {
                    (Square::H1, Square::F1)
                } else {
                    (Square::H8, Square::F8)
                };
                self.move_piece(rook_to, rook_from, Piece::Rook, us);
            }
            MoveFlag::QueenCastle => {
                self.move_piece(to, from, Piece::King, us);
                let (rook_from, rook_to) = if us == Color::White {
                    (Square::A1, Square::D1)
                } else {
                    (Square::A8, Square::D8)
                };
                self.move_piece(rook_to, rook_from, Piece::Rook, us);
            }
            _ if flag.is_promotion() => {
                let promo_piece = flag.promotion_piece().unwrap();
                self.remove_piece(to, promo_piece, us);
                self.add_piece(from, Piece::Pawn, us);
                if let Some(cap_piece) = undo.captured {
                    self.add_piece(to, cap_piece, them);
                }
            }
            _ => {}
        }

        // Restore state
        self.castling = undo.castling;
        self.ep_square = undo.ep_square;
        self.halfmove_clock = undo.halfmove_clock;
        self.hash = undo.hash;
        self.checkers = undo.checkers;
        
        // Fullmove number
        if us == Color::Black {
            self.fullmove_number -= 1;
        }
    }

    /// Unmake a move fast (for perft - doesn't restore hash/clock).
    #[inline(always)]
    pub fn unmake_move_fast(&mut self, mv: &Move, undo: UndoInfo) {
        let from = mv.from();
        let to = mv.to();
        let flag = mv.flag();
        
        // Switch side back
        self.turn = !self.turn;
        let us = self.turn;
        let them = !us;

        // Find the moving piece
        let piece = if flag.is_promotion() {
            Piece::Pawn
        } else {
            self.piece_at(to).map(|(p, _)| p).unwrap_or(Piece::Pawn)
        };

        match flag {
            MoveFlag::Quiet | MoveFlag::DoublePawnPush => {
                self.move_piece_fast(to, from, piece, us);
            }
            MoveFlag::Capture => {
                self.move_piece_fast(to, from, piece, us);
                if let Some(cap_piece) = undo.captured {
                    self.add_piece_fast(to, cap_piece, them);
                }
            }
            MoveFlag::EnPassant => {
                self.move_piece_fast(to, from, Piece::Pawn, us);
                let cap_sq = if us == Color::White {
                    unsafe { Square::from_index_unchecked(to.index() - 8) }
                } else {
                    unsafe { Square::from_index_unchecked(to.index() + 8) }
                };
                self.add_piece_fast(cap_sq, Piece::Pawn, them);
            }
            MoveFlag::KingCastle => {
                self.move_piece_fast(to, from, Piece::King, us);
                let (rook_from, rook_to) = if us == Color::White {
                    (Square::H1, Square::F1)
                } else {
                    (Square::H8, Square::F8)
                };
                self.move_piece_fast(rook_to, rook_from, Piece::Rook, us);
            }
            MoveFlag::QueenCastle => {
                self.move_piece_fast(to, from, Piece::King, us);
                let (rook_from, rook_to) = if us == Color::White {
                    (Square::A1, Square::D1)
                } else {
                    (Square::A8, Square::D8)
                };
                self.move_piece_fast(rook_to, rook_from, Piece::Rook, us);
            }
            _ if flag.is_promotion() => {
                let promo_piece = flag.promotion_piece().unwrap();
                self.remove_piece_fast(to, promo_piece, us);
                self.add_piece_fast(from, Piece::Pawn, us);
                if let Some(cap_piece) = undo.captured {
                    self.add_piece_fast(to, cap_piece, them);
                }
            }
            _ => {}
        }

        // Restore minimal state needed for move generation
        self.castling = undo.castling;
        self.ep_square = undo.ep_square;
        self.checkers = undo.checkers;
    }

    /// Make a null move (pass the turn without making any move).
    /// Used for null move pruning in search.
    /// Returns a new board with side to move switched.
    #[inline]
    pub fn make_null_move(&self) -> Board {
        let mut result = *self;
        
        // Clear en passant
        result.ep_square = None;
        
        // Switch side
        result.turn = !self.turn;
        
        // Update hash for side change
        result.hash ^= ZOBRIST.side();
        
        // Recalculate checkers for new side
        result.update_checkers();
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_unmake_quiet() {
        let mut board = Board::startpos();
        let initial_fen = board.to_fen();
        let initial_hash = board.hash();
        
        // e2-e4
        let mv = Move::new(Square::E2, Square::E4, MoveFlag::DoublePawnPush);
        let undo = board.make_move(mv);
        
        assert!(board.piece_at(Square::E4).is_some());
        assert!(board.piece_at(Square::E2).is_none());
        
        board.unmake_move(mv, undo);
        
        assert_eq!(board.to_fen(), initial_fen);
        assert_eq!(board.hash(), initial_hash);
    }
}

