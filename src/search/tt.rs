//! Transposition Table for caching search results.
//!
//! This module provides a high-performance, lock-free transposition table
//! that stores search results to avoid redundant computation.
//!
//! # Design
//! - 8-byte entries packed into AtomicU64 for lock-free access
//! - Depth-preferred replacement with age-based eviction
//! - Lock-free for Lazy SMP multi-threading support

use crate::types::{Depth, Hash, Move, Score};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

/// Type of bound stored in TT entry
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BoundType {
    /// No bound (empty entry)
    None = 0,
    /// Exact score (PV node)
    Exact = 1,
    /// Lower bound (fail-high, score >= beta)
    LowerBound = 2,
    /// Upper bound (fail-low, score <= alpha)
    UpperBound = 3,
}

impl From<u8> for BoundType {
    fn from(v: u8) -> Self {
        match v & 0x03 {
            1 => BoundType::Exact,
            2 => BoundType::LowerBound,
            3 => BoundType::UpperBound,
            _ => BoundType::None,
        }
    }
}

pub struct TTBucket {
    hash_key: AtomicU64,
    data: AtomicU64,
}

impl Default for TTBucket {
    fn default() -> Self {
        Self { hash_key: AtomicU64::new(0), data: AtomicU64::new(0) }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TTEntry {
    best_move: u16,
    score: i16,
    depth: i8,
    bound_and_age: u8,
}

impl TTEntry {
    pub fn new(m: Option<Move>, s: Score, d: Depth, b: BoundType, gen: u8) -> Self {
        Self {
            best_move: encode_move(m),
            score: s.raw() as i16,
            depth: d.raw() as i8,
            bound_and_age: (b as u8) | ((gen & 0x3F) << 2),
        }
    }

    #[inline]
    pub fn to_data(&self) -> u64 {
        ((self.best_move as u64) << 32)
            | (((self.score as u16) as u64) << 16)
            | ((self.depth as u8 as u64) << 8)
            | (self.bound_and_age as u64)
    }

    #[inline]
    pub fn from_data(r: u64) -> Self {
        Self {
            best_move: (r >> 32) as u16,
            score: (r >> 16) as i16,
            depth: (r >> 8) as i8,
            bound_and_age: r as u8,
        }
    }

    #[inline] pub fn bound(&self) -> BoundType { BoundType::from(self.bound_and_age) }
    #[inline] pub fn generation(&self) -> u8 { self.bound_and_age >> 2 }
    #[inline] pub fn score(&self) -> Score { Score::cp(self.score as i32) }
    #[inline] pub fn depth(&self) -> Depth { Depth::new(self.depth as i32) }
    #[inline] pub fn best_move(&self) -> Option<Move> { decode_move(self.best_move) }
    #[inline] pub fn is_empty(&self) -> bool { self.bound() == BoundType::None }
}

fn encode_move(m: Option<Move>) -> u16 {
    match m { Some(mv) => mv.bits(), None => 0 }
}

fn decode_move(e: u16) -> Option<Move> {
    if e == 0 { None } else { Some(Move::from_bits(e)) }
}

pub struct TranspositionTable {
    entries: Vec<TTBucket>,
    generation: AtomicU8,
    size_mb: usize,
}

unsafe impl Send for TranspositionTable {}
unsafe impl Sync for TranspositionTable {}

impl TranspositionTable {
    pub fn new(mb: usize) -> Self {
        let n = ((mb * 1024 * 1024) / 16).next_power_of_two() / 2;
        let n = n.max(1024);
        let mut entries = Vec::with_capacity(n);
        for _ in 0..n { entries.push(TTBucket::default()); }
        Self { entries, generation: AtomicU8::new(0), size_mb: mb }
    }

    #[inline] pub fn len(&self) -> usize { self.entries.len() }
    #[inline] pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    #[inline] pub fn size_mb(&self) -> usize { self.size_mb }
    #[inline] pub fn generation(&self) -> u8 { self.generation.load(Ordering::Relaxed) }
    pub fn new_search(&self) { self.generation.fetch_add(1, Ordering::Relaxed); }
    #[inline] fn index(&self, h: Hash) -> usize { (h as usize) & (self.entries.len() - 1) }

    #[inline]
    pub fn probe(&self, h: Hash) -> Option<TTEntry> {
        let b = &self.entries[self.index(h)];
        let d = b.data.load(Ordering::Relaxed);
        let k = b.hash_key.load(Ordering::Relaxed);
        if (k ^ d) == h {
            let e = TTEntry::from_data(d);
            if !e.is_empty() { return Some(e); }
        }
        None
    }

    pub fn store(&self, h: Hash, m: Option<Move>, s: Score, d: Depth, b: BoundType) {
        let i = self.index(h);
        let bucket = &self.entries[i];
        let ed = bucket.data.load(Ordering::Relaxed);
        let ek = bucket.hash_key.load(Ordering::Relaxed);
        let gen = self.generation();
        let e = if (ek ^ ed) == h { TTEntry::from_data(ed) } else { TTEntry::default() };

        if e.is_empty() || (ek ^ ed) != h || e.generation() != gen || d.raw() >= e.depth.into() {
            let m = m.or_else(|| e.best_move());
            let ne = TTEntry::new(m, s, d, b, gen);
            let nd = ne.to_data();
            bucket.data.store(nd, Ordering::Relaxed);
            bucket.hash_key.store(h ^ nd, Ordering::Relaxed);
        }
    }

    pub fn clear(&self) {
        for b in &self.entries {
            b.data.store(0, Ordering::Relaxed);
            b.hash_key.store(0, Ordering::Relaxed);
        }
        self.generation.store(0, Ordering::Relaxed);
    }

    pub fn hashfull(&self) -> u32 {
        let gen = self.generation();
        let s = self.entries.len().min(1000);
        let u = self.entries[..s].iter().filter(|b| {
            let e = TTEntry::from_data(b.data.load(Ordering::Relaxed));
            !e.is_empty() && e.generation() == gen
        }).count();
        ((u * 1000) / s) as u32
    }

    #[inline]
    pub fn prefetch(&self, h: Hash) {
        let i = self.index(h);
        let p = self.entries.as_ptr().wrapping_add(i) as *const i8;
        #[cfg(target_arch = "x86_64")] unsafe { std::arch::x86_64::_mm_prefetch(p, std::arch::x86_64::_MM_HINT_T0); }
        #[cfg(target_arch = "x86")] unsafe { std::arch::x86::_mm_prefetch(p, std::arch::x86::_MM_HINT_T0); }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))] let _ = p;
    }
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::new(16) // 16 MB default
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MoveFlag, Square};

    #[test]
    fn test_tt_basic() {
        let tt = TranspositionTable::new(1);
        let hash: Hash = 0x123456789ABCDEF0;

        // Initially empty
        assert!(tt.probe(hash).is_none());

        // Store and retrieve
        tt.store(hash, None, Score::cp(100), Depth::new(5), BoundType::Exact);

        let entry = tt.probe(hash).expect("Entry should exist");
        assert_eq!(entry.score().raw(), 100);
        assert_eq!(entry.depth().raw(), 5);
        assert_eq!(entry.bound(), BoundType::Exact);
    }

    #[test]
    fn test_move_encoding() {
        let mv = Move::new(Square::E2, Square::E4, MoveFlag::DoublePawnPush);
        let encoded = encode_move(Some(mv));
        let decoded = decode_move(encoded).unwrap();
        assert_eq!(mv.from(), decoded.from());
        assert_eq!(mv.to(), decoded.to());
    }

    #[test]
    fn test_entry_pack_unpack() {
        let entry = TTEntry::new(
            None,
            Score::cp(150),
            Depth::new(8),
            BoundType::LowerBound,
            5,
        );

        let packed = entry.to_data();
        let unpacked = TTEntry::from_data(packed);

        assert_eq!(entry.score, unpacked.score);
        assert_eq!(entry.depth, unpacked.depth);
        assert_eq!(entry.bound(), unpacked.bound());
        assert_eq!(entry.generation(), unpacked.generation());
    }
}
