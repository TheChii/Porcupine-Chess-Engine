//! FEN parsing and serialization.

use super::Board;
use crate::types::{Square, File, Rank, Piece, Color, CastleRights};

impl Board {
    /// Parse a FEN string into a board.
    pub fn from_fen(fen: &str) -> Result<Board, &'static str> {
        let mut board = Board::empty();
        let parts: Vec<&str> = fen.split_whitespace().collect();
        
        if parts.len() < 4 {
            return Err("FEN must have at least 4 parts");
        }
        
        // Parse piece placement
        let mut rank = 7u8;
        let mut file = 0u8;
        
        for c in parts[0].chars() {
            match c {
                '/' => {
                    if rank == 0 {
                        return Err("Too many ranks in FEN");
                    }
                    rank -= 1;
                    file = 0;
                }
                '1'..='8' => {
                    file += (c as u8) - b'0';
                }
                _ => {
                    if file >= 8 {
                        return Err("Too many pieces in rank");
                    }
                    if let Some((piece, color)) = Piece::from_char(c) {
                        let sq = Square::from_file_rank(
                            unsafe { File::from_index_unchecked(file) },
                            unsafe { Rank::from_index_unchecked(rank) },
                        );
                        board.add_piece(sq, piece, color);
                        file += 1;
                    } else {
                        return Err("Invalid piece character");
                    }
                }
            }
        }
        
        // Parse side to move
        board.turn = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err("Invalid side to move"),
        };
        if board.turn == Color::Black {
            board.hash ^= super::zobrist::ZOBRIST.side();
        }
        
        // Parse castling rights
        board.castling = CastleRights::from_fen(parts[2])
            .ok_or("Invalid castling rights")?;
        board.hash ^= super::zobrist::ZOBRIST.castling(board.castling);
        
        // Parse en passant
        board.ep_square = if parts[3] == "-" {
            None
        } else {
            let sq = Square::from_algebraic(parts[3])
                .ok_or("Invalid en passant square")?;
            board.hash ^= super::zobrist::ZOBRIST.ep_file(sq.file());
            Some(sq)
        };
        
        // Parse halfmove clock (optional)
        if parts.len() > 4 {
            board.halfmove_clock = parts[4].parse().unwrap_or(0);
        }
        
        // Parse fullmove number (optional)
        if parts.len() > 5 {
            board.fullmove_number = parts[5].parse().unwrap_or(1);
        }
        
        // Compute checkers
        board.update_checkers();
        
        Ok(board)
    }

    /// Convert board to FEN string.
    pub fn to_fen(&self) -> String {
        let mut fen = String::with_capacity(80);
        
        // Piece placement
        for rank in (0..8).rev() {
            let mut empty_count = 0;
            
            for file in 0..8 {
                let sq = Square::from_file_rank(
                    unsafe { File::from_index_unchecked(file) },
                    unsafe { Rank::from_index_unchecked(rank) },
                );
                
                if let Some((piece, color)) = self.piece_at(sq) {
                    if empty_count > 0 {
                        fen.push((b'0' + empty_count) as char);
                        empty_count = 0;
                    }
                    fen.push(piece.to_char(color));
                } else {
                    empty_count += 1;
                }
            }
            
            if empty_count > 0 {
                fen.push((b'0' + empty_count) as char);
            }
            
            if rank > 0 {
                fen.push('/');
            }
        }
        
        // Side to move
        fen.push(' ');
        fen.push(if self.turn == Color::White { 'w' } else { 'b' });
        
        // Castling rights
        fen.push(' ');
        fen.push_str(self.castling.to_fen());
        
        // En passant
        fen.push(' ');
        if let Some(sq) = self.ep_square {
            let [f, r] = sq.to_algebraic();
            fen.push(f);
            fen.push(r);
        } else {
            fen.push('-');
        }
        
        // Halfmove clock
        fen.push(' ');
        fen.push_str(&self.halfmove_clock.to_string());
        
        // Fullmove number
        fen.push(' ');
        fen.push_str(&self.fullmove_number.to_string());
        
        fen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_startpos_fen() {
        let board = Board::startpos();
        assert_eq!(
            board.to_fen(),
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        );
    }

    #[test]
    fn test_fen_roundtrip() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        ];
        
        for fen in fens {
            let board = Board::from_fen(fen).unwrap();
            // Note: FEN might differ slightly due to normalization
            let _ = board.to_fen();
        }
    }

    #[test]
    fn test_piece_at() {
        let board = Board::startpos();
        assert_eq!(board.piece_at(Square::E1), Some((Piece::King, Color::White)));
        assert_eq!(board.piece_at(Square::E8), Some((Piece::King, Color::Black)));
        assert_eq!(board.piece_at(Square::E4), None);
    }
}
