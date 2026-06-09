//! Move type and MoveList container.

use core::fmt;
use crate::types::{Square, Piece};

/// Move flags for special move types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MoveFlag {
    Quiet = 0,
    DoublePawnPush = 1,
    KingCastle = 2,
    QueenCastle = 3,
    Capture = 4,
    EnPassant = 5,
    // 6, 7 unused
    PromoKnight = 8,
    PromoBishop = 9,
    PromoRook = 10,
    PromoQueen = 11,
    PromoKnightCapture = 12,
    PromoBishopCapture = 13,
    PromoRookCapture = 14,
    PromoQueenCapture = 15,
}

impl MoveFlag {
    /// Create from raw value.
    #[inline(always)]
    pub const fn from_u8(val: u8) -> MoveFlag {
        unsafe { core::mem::transmute(val & 0xF) }
    }

    /// Check if this is a capture.
    #[inline(always)]
    pub const fn is_capture(self) -> bool {
        (self as u8 & 4) != 0 || self as u8 == 5 // Capture flag or EP
    }

    /// Check if this is a promotion.
    #[inline(always)]
    pub const fn is_promotion(self) -> bool {
        (self as u8 & 8) != 0
    }

    /// Get the promotion piece type (if this is a promotion).
    #[inline(always)]
    pub const fn promotion_piece(self) -> Option<Piece> {
        if self.is_promotion() {
            Some(Piece::from_promotion_index((self as u8) & 3))
        } else {
            None
        }
    }

    /// Create a promotion flag.
    #[inline(always)]
    pub const fn promotion(piece: Piece, capture: bool) -> MoveFlag {
        let base = match piece {
            Piece::Knight => 8,
            Piece::Bishop => 9,
            Piece::Rook => 10,
            Piece::Queen => 11,
            _ => 11, // Default to queen
        };
        MoveFlag::from_u8(base + if capture { 4 } else { 0 })
    }
}

/// A packed 16-bit chess move.
///
/// Layout:
/// - Bits 0-5: Source square (0-63)
/// - Bits 6-11: Destination square (0-63)
/// - Bits 12-15: Move flag
#[derive(Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct Move(u16);

impl Move {
    /// Null move constant.
    pub const NULL: Move = Move(0);

    /// Create a new move.
    #[inline(always)]
    pub const fn new(from: Square, to: Square, flag: MoveFlag) -> Move {
        Move(
            (from.index() as u16)
                | ((to.index() as u16) << 6)
                | ((flag as u16) << 12)
        )
    }

    /// Get the source square.
    #[inline(always)]
    pub const fn from(self) -> Square {
        unsafe { Square::from_index_unchecked((self.0 & 0x3F) as u8) }
    }

    /// Get the destination square.
    #[inline(always)]
    pub const fn to(self) -> Square {
        unsafe { Square::from_index_unchecked(((self.0 >> 6) & 0x3F) as u8) }
    }

    /// Get the move flag.
    #[inline(always)]
    pub const fn flag(self) -> MoveFlag {
        MoveFlag::from_u8((self.0 >> 12) as u8)
    }

    /// Check if this is a null move.
    #[inline(always)]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    /// Check if this is a capture.
    #[inline(always)]
    pub const fn is_capture(self) -> bool {
        self.flag().is_capture()
    }

    /// Check if this is a promotion.
    #[inline(always)]
    pub const fn is_promotion(self) -> bool {
        self.flag().is_promotion()
    }

    /// Get raw bits.
    #[inline(always)]
    pub const fn bits(self) -> u16 {
        self.0
    }

    /// Create from raw bits.
    #[inline(always)]
    pub const fn from_bits(bits: u16) -> Move {
        Move(bits)
    }

    /// Convert to UCI string.
    pub fn to_uci(self) -> String {
        let from = self.from().to_algebraic();
        let to = self.to().to_algebraic();
        let mut s = String::with_capacity(5);
        s.push(from[0]);
        s.push(from[1]);
        s.push(to[0]);
        s.push(to[1]);
        
        if let Some(promo) = self.flag().promotion_piece() {
            s.push(match promo {
                Piece::Knight => 'n',
                Piece::Bishop => 'b',
                Piece::Rook => 'r',
                Piece::Queen => 'q',
                _ => 'q',
            });
        }
        
        s
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Move({})", self.to_uci())
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

/// A scored move for move ordering.
#[derive(Clone, Copy)]
pub struct ScoredMove {
    pub mv: Move,
    pub score: i16,
}

impl ScoredMove {
    #[inline(always)]
    pub const fn new(mv: Move, score: i16) -> ScoredMove {
        ScoredMove { mv, score }
    }
}

/// Fixed-size move list (stack allocated).
pub struct MoveList {
    moves: [Move; 256],
    count: usize,
}

impl MoveList {
    /// Create an empty move list.
    #[inline]
    pub const fn new() -> MoveList {
        MoveList {
            moves: [Move::NULL; 256],
            count: 0,
        }
    }

    /// Get the number of moves.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Check if empty.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Push a move.
    #[inline(always)]
    pub fn push(&mut self, mv: Move) {
        debug_assert!(self.count < 256);
        self.moves[self.count] = mv;
        self.count += 1;
    }

    /// Get a move by index.
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<Move> {
        if index < self.count {
            Some(self.moves[index])
        } else {
            None
        }
    }

    /// Get a move by index, unchecked.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> Move {
        *self.moves.get_unchecked(index)
    }

    /// Clear the list.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.count = 0;
    }

    /// Iterate over moves.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = Move> + '_ {
        self.moves[..self.count].iter().copied()
    }

    /// Get as a slice.
    #[inline(always)]
    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.count]
    }

    /// Get as a mutable slice.
    #[inline(always)]
    pub fn as_slice_mut(&mut self) -> &mut [Move] {
        &mut self.moves[..self.count]
    }

    /// Check if a move is in the list.
    pub fn contains(&self, mv: Move) -> bool {
        self.iter().any(|m| m.bits() == mv.bits())
    }
}

impl Default for MoveList {
    fn default() -> MoveList {
        MoveList::new()
    }
}

impl IntoIterator for MoveList {
    type Item = Move;
    type IntoIter = core::iter::Take<core::array::IntoIter<Move, 256>>;

    fn into_iter(self) -> Self::IntoIter {
        self.moves.into_iter().take(self.count)
    }
}

impl<'a> IntoIterator for &'a MoveList {
    type Item = Move;
    type IntoIter = core::iter::Copied<core::slice::Iter<'a, Move>>;

    fn into_iter(self) -> Self::IntoIter {
        self.moves[..self.count].iter().copied()
    }
}

impl fmt::Debug for MoveList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// Trait for move collection (allows bulk counting without storing).
pub trait MoveSink {
    fn push(&mut self, mv: Move);
}

impl MoveSink for MoveList {
    #[inline(always)]
    fn push(&mut self, mv: Move) {
        self.push(mv);
    }
}

/// A sink that just counts moves.
pub struct MoveCounter {
    pub count: u64,
}

impl MoveCounter {
    pub fn new() -> Self {
        Self { count: 0 }
    }
}

impl MoveSink for MoveCounter {
    #[inline(always)]
    fn push(&mut self, _mv: Move) {
        self.count += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_move_encoding() {
        let mv = Move::new(Square::E2, Square::E4, MoveFlag::DoublePawnPush);
        assert_eq!(mv.from(), Square::E2);
        assert_eq!(mv.to(), Square::E4);
        assert_eq!(mv.flag(), MoveFlag::DoublePawnPush);
    }

    #[test]
    fn test_move_uci() {
        let mv = Move::new(Square::E2, Square::E4, MoveFlag::DoublePawnPush);
        assert_eq!(mv.to_uci(), "e2e4");

        let promo = Move::new(Square::E7, Square::E8, MoveFlag::PromoQueen);
        assert_eq!(promo.to_uci(), "e7e8q");
    }

    #[test]
    fn test_movelist() {
        let mut list = MoveList::new();
        assert!(list.is_empty());

        list.push(Move::new(Square::E2, Square::E4, MoveFlag::DoublePawnPush));
        assert_eq!(list.len(), 1);

        list.push(Move::new(Square::D2, Square::D4, MoveFlag::DoublePawnPush));
        assert_eq!(list.len(), 2);
    }
}
