//! Board state representation.

mod fen;
mod zobrist;
mod make_move;


pub use zobrist::ZOBRIST;

use crate::bitboard::Bitboard;
use crate::types::{Square, Piece, Color, CastleRights};

/// Chess board state.
#[derive(Clone, Copy)]
pub struct Board {
    /// Bitboards for each piece type.
    pieces: [Bitboard; 6],
    /// Bitboards for each color.
    colors: [Bitboard; 2],
    /// Side to move.
    turn: Color,
    /// Castling rights.
    castling: CastleRights,
    /// En passant target square (if any).
    ep_square: Option<Square>,
    /// Halfmove clock for 50-move rule.
    halfmove_clock: u8,
    /// Fullmove number.
    fullmove_number: u16,
    /// Zobrist hash.
    hash: u64,
    /// Cached checkers bitboard.
    checkers: Bitboard,
}

impl Board {
    /// Create a new empty board.
    pub const fn empty() -> Board {
        Board {
            pieces: [Bitboard::EMPTY; 6],
            colors: [Bitboard::EMPTY; 2],
            turn: Color::White,
            castling: CastleRights::NONE,
            ep_square: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            hash: 0,
            checkers: Bitboard::EMPTY,
        }
    }

    /// Get the starting position.
    pub fn startpos() -> Board {
        Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap()
    }

    /// Get a reference to the piece bitboards.
    #[inline(always)]
    pub fn pieces(&self) -> &[Bitboard; 6] {
        &self.pieces
    }

    /// Get a reference to the color bitboards.
    #[inline(always)]
    pub fn colors(&self) -> &[Bitboard; 2] {
        &self.colors
    }

    /// Get the side to move.
    #[inline(always)]
    pub fn turn(&self) -> Color {
        self.turn
    }

    /// Get castling rights.
    #[inline(always)]
    pub fn castling(&self) -> CastleRights {
        self.castling
    }

    /// Get en passant square.
    #[inline(always)]
    pub fn ep_square(&self) -> Option<Square> {
        self.ep_square
    }

    /// Get halfmove clock.
    #[inline(always)]
    pub fn halfmove_clock(&self) -> u8 {
        self.halfmove_clock
    }

    /// Get fullmove number.
    #[inline(always)]
    pub fn fullmove_number(&self) -> u16 {
        self.fullmove_number
    }

    /// Get the Zobrist hash.
    #[inline(always)]
    pub fn hash(&self) -> u64 {
        self.hash
    }

    /// Get a pawn structure hash (for correction history).
    /// This combines info about pawn positions for both sides.
    #[inline]
    pub fn pawn_hash(&self) -> u64 {
        // Simple pawn hash: XOR of pawn bitboard bits with a mixing constant
        let white_pawns = self.piece_color_bb(Piece::Pawn, Color::White).0;
        let black_pawns = self.piece_color_bb(Piece::Pawn, Color::Black).0;
        white_pawns.wrapping_mul(0x9E3779B97F4A7C15) ^ black_pawns.wrapping_mul(0xC6A4A7935BD1E995)
    }

    /// Get bitboard of pieces that are giving check.
    #[inline(always)]
    pub fn checkers(&self) -> Bitboard {
        self.checkers
    }

    /// Check if the side to move is in check.
    #[inline(always)]
    pub fn in_check(&self) -> bool {
        self.checkers.any()
    }

    /// Get bitboard of all occupied squares.
    #[inline(always)]
    pub fn occupied(&self) -> Bitboard {
        self.colors[0] | self.colors[1]
    }

    /// Get bitboard of empty squares.
    #[inline(always)]
    pub fn empty_squares(&self) -> Bitboard {
        !self.occupied()
    }

    /// Get bitboard for a specific piece type.
    #[inline(always)]
    pub fn piece_bb(&self, piece: Piece) -> Bitboard {
        self.pieces[piece.index()]
    }

    /// Get bitboard for a specific color.
    #[inline(always)]
    pub fn color_bb(&self, color: Color) -> Bitboard {
        self.colors[color.index()]
    }

    /// Get bitboard for a specific piece and color.
    #[inline(always)]
    pub fn piece_color_bb(&self, piece: Piece, color: Color) -> Bitboard {
        self.pieces[piece.index()] & self.colors[color.index()]
    }

    /// Get our pieces bitboard.
    #[inline(always)]
    pub fn us(&self) -> Bitboard {
        self.colors[self.turn.index()]
    }

    /// Get their pieces bitboard.
    #[inline(always)]
    pub fn them(&self) -> Bitboard {
        self.colors[(!self.turn).index()]
    }

    /// Find the king square for a color.
    #[inline]
    pub fn king_square(&self, color: Color) -> Square {
        let king_bb = self.piece_color_bb(Piece::King, color);
        debug_assert!(king_bb.exactly_one());
        unsafe { king_bb.lsb_unchecked() }
    }

    /// Get the piece at a square.
    /// Get the piece at a square.
    pub fn piece_at(&self, sq: Square) -> Option<(Piece, Color)> {
        let sq_bb = Bitboard::from_square(sq);
        
        // Fast fail using computed occupied
        if ((self.colors[0] | self.colors[1]) & sq_bb).is_empty() {
             return None;
        }

        let color = if (self.colors[0] & sq_bb).any() {
            Color::White
        } else {
            Color::Black
        };
        
        // Unrolled piece check (ordered by frequency)
        if (self.pieces[0] & sq_bb).any() { return Some((Piece::Pawn, color)); }
        if (self.pieces[1] & sq_bb).any() { return Some((Piece::Knight, color)); }
        if (self.pieces[2] & sq_bb).any() { return Some((Piece::Bishop, color)); }
        if (self.pieces[3] & sq_bb).any() { return Some((Piece::Rook, color)); }
        if (self.pieces[4] & sq_bb).any() { return Some((Piece::Queen, color)); }
        return Some((Piece::King, color));
    }

    /// Add a piece to the board.
    #[inline]
    pub fn add_piece(&mut self, sq: Square, piece: Piece, color: Color) {
        let sq_bb = Bitboard::from_square(sq);
        self.pieces[piece.index()] |= sq_bb;
        self.colors[color.index()] |= sq_bb;
        self.hash ^= ZOBRIST.piece_square(piece, color, sq);
    }

    /// Remove a piece from the board.
    #[inline]
    pub fn remove_piece(&mut self, sq: Square, piece: Piece, color: Color) {
        let sq_bb = Bitboard::from_square(sq);
        self.pieces[piece.index()] &= !sq_bb;
        self.colors[color.index()] &= !sq_bb;
        self.hash ^= ZOBRIST.piece_square(piece, color, sq);
    }

    /// Move a piece on the board.
    #[inline]
    pub fn move_piece(&mut self, from: Square, to: Square, piece: Piece, color: Color) {
        let from_to = Bitboard::from_square(from) | Bitboard::from_square(to);
        self.pieces[piece.index()] ^= from_to;
        self.colors[color.index()] ^= from_to;
        self.hash ^= ZOBRIST.piece_square(piece, color, from);
        self.hash ^= ZOBRIST.piece_square(piece, color, to);
    }

    /// Add a piece without updating hash (fast path for perft).
    #[inline(always)]
    pub fn add_piece_fast(&mut self, sq: Square, piece: Piece, color: Color) {
        let sq_bb = Bitboard::from_square(sq);
        self.pieces[piece.index()] |= sq_bb;
        self.colors[color.index()] |= sq_bb;
    }

    /// Remove a piece without updating hash (fast path for perft).
    #[inline(always)]
    pub fn remove_piece_fast(&mut self, sq: Square, piece: Piece, color: Color) {
        let sq_bb = Bitboard::from_square(sq);
        self.pieces[piece.index()] &= !sq_bb;
        self.colors[color.index()] &= !sq_bb;
    }

    /// Move a piece without updating hash (fast path for perft).
    #[inline(always)]
    pub fn move_piece_fast(&mut self, from: Square, to: Square, piece: Piece, color: Color) {
        let from_to = Bitboard::from_square(from) | Bitboard::from_square(to);
        self.pieces[piece.index()] ^= from_to;
        self.colors[color.index()] ^= from_to;
    }

    /// Compute attackers to a square.
    pub fn attackers_to(&self, sq: Square, occ: Bitboard) -> Bitboard {
        use crate::attacks::{pawn_attacks, knight_attacks, king_attacks, bishop_attacks, rook_attacks};
        
        let bishops = self.pieces[Piece::Bishop.index()] | self.pieces[Piece::Queen.index()];
        let rooks = self.pieces[Piece::Rook.index()] | self.pieces[Piece::Queen.index()];
        
        (pawn_attacks(Color::Black, sq) & self.piece_color_bb(Piece::Pawn, Color::White))
        | (pawn_attacks(Color::White, sq) & self.piece_color_bb(Piece::Pawn, Color::Black))
        | (knight_attacks(sq) & self.pieces[Piece::Knight.index()])
        | (king_attacks(sq) & self.pieces[Piece::King.index()])
        | (bishop_attacks(sq, occ) & bishops)
        | (rook_attacks(sq, occ) & rooks)
    }

    /// Compute checkers for the side to move.
    pub fn compute_checkers(&self) -> Bitboard {
        let king_sq = self.king_square(self.turn);
        self.attackers_to(king_sq, self.occupied()) & self.them()
    }

    /// Update cached checkers.
    pub fn update_checkers(&mut self) {
        self.checkers = self.compute_checkers();
    }
}

impl Default for Board {
    fn default() -> Board {
        Board::startpos()
    }
}

impl core::fmt::Debug for Board {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "Board {{")?;
        writeln!(f, "  FEN: {}", self.to_fen())?;
        writeln!(f, "  Turn: {:?}", self.turn)?;
        writeln!(f, "  Castling: {}", self.castling)?;
        writeln!(f, "  EP: {:?}", self.ep_square)?;
        writeln!(f, "  Hash: 0x{:016X}", self.hash)?;
        write!(f, "}}")
    }
}
