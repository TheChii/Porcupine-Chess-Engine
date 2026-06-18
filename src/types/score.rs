//! Score type for search.
//!
//! Handles regular centipawn scores, mate scores, draws, and special values.
//! Optimized for alpha-beta search with proper mate score handling.

use std::fmt;
use std::ops::{Add, Neg, Sub};

/// Special score values
pub const SCORE_NONE: i32 = -32001;
pub const SCORE_INFINITY: i32 = 32000;
pub const SCORE_MATE: i32 = 31000;
pub const SCORE_DRAW: i32 = 0;

// Mate score bounds for detection
const SCORE_MATE_IN_MAX: i32 = SCORE_MATE - 1000;
const SCORE_MATED_IN_MAX: i32 = -SCORE_MATE + 1000;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct Score(pub i16);

impl Score {
    #[inline] pub const fn cp(v: i32) -> Self { Score(v as i16) }
    #[inline] pub const fn mate_in(p: i32) -> Self { Score((SCORE_MATE - p) as i16) }
    #[inline] pub const fn mated_in(p: i32) -> Self { Score((-SCORE_MATE + p) as i16) }
    #[inline] pub const fn draw() -> Self { Score(SCORE_DRAW as i16) }
    #[inline] pub const fn infinity() -> Self { Score(SCORE_INFINITY as i16) }
    #[inline] pub const fn neg_infinity() -> Self { Score((-SCORE_INFINITY) as i16) }
    #[inline] pub const fn none() -> Self { Score(SCORE_NONE as i16) }
    #[inline] pub const fn raw(self) -> i32 { self.0 as i32 }
    #[inline] pub const fn is_mate(self) -> bool { self.0 as i32 >= SCORE_MATE_IN_MAX }
    #[inline] pub const fn is_mated(self) -> bool { self.0 as i32 <= SCORE_MATED_IN_MAX }
    #[inline] pub const fn is_mate_score(self) -> bool { self.is_mate() || self.is_mated() }

    #[inline]
    pub const fn mate_distance(self) -> Option<i32> {
        if self.is_mate() { Some(SCORE_MATE - self.0 as i32) }
        else if self.is_mated() { Some(self.0 as i32 + SCORE_MATE) }
        else { None }
    }

    #[inline]
    pub const fn to_tt(self, p: i32) -> Self {
        if self.is_mate() { Score((self.0 as i32 + p) as i16) }
        else if self.is_mated() { Score((self.0 as i32 - p) as i16) }
        else { self }
    }

    #[inline]
    pub const fn from_tt(self, p: i32) -> Self {
        if self.is_mate() { Score((self.0 as i32 - p) as i16) }
        else if self.is_mated() { Score((self.0 as i32 + p) as i16) }
        else { self }
    }

    #[inline] pub const fn add_const(self, v: i32) -> Self { Score(self.0 + v as i16) }
    #[inline] pub const fn sub_const(self, v: i32) -> Self { Score(self.0 - v as i16) }
}

impl Add for Score {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Score(self.0 + rhs.0)
    }
}

impl Sub for Score {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Score(self.0 - rhs.0)
    }
}

impl Neg for Score {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Score(-self.0)
    }
}

impl From<i32> for Score {
    #[inline]
    fn from(v: i32) -> Self {
        Score(v as i16)
    }
}

impl fmt::Display for Score {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_mate() {
            let moves = (SCORE_MATE - self.0 as i32 + 1) / 2;
            write!(f, "mate {}", moves)
        } else if self.is_mated() {
            let moves = (self.0 as i32 + SCORE_MATE + 1) / 2;
            write!(f, "mate -{}", moves)
        } else {
            write!(f, "cp {}", self.0)
        }
    }
}

impl fmt::Debug for Score {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Score({})", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mate_scores() {
        let mate_in_3 = Score::mate_in(5); // 5 ply = mate in 3 moves
        assert!(mate_in_3.is_mate());
        assert!(!mate_in_3.is_mated());
        assert_eq!(mate_in_3.mate_distance(), Some(5));

        let mated_in_2 = Score::mated_in(3); // 3 ply = mated in 2 moves
        assert!(!mated_in_2.is_mate());
        assert!(mated_in_2.is_mated());
        assert_eq!(mated_in_2.mate_distance(), Some(3));
    }

    #[test]
    fn test_tt_adjustment() {
        let mate = Score::mate_in(5);
        let tt_score = mate.to_tt(2);
        let restored = tt_score.from_tt(2);
        assert_eq!(mate, restored);
    }
}
