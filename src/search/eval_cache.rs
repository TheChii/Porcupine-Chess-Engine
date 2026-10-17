//! Lockless Static Evaluation Cache (EvalCache).
//!
//! Stores computed static evaluations by Zobrist hash to bypass dense NNUE
//! forward passes on transpositions across negamax search and quiescence search.
//!
//! # Design
//! - 1,048,576 entries (8.0 MB heap footprint).
//! - Packed 64-bit atomic integer per entry:
//!   - High 48 bits: `hash >> 16` (tag for collision check)
//!   - Low 16 bits: `score as i16 as u16` (evaluation in centipawns)
//! - Single-instruction relaxed atomic load/store (`mov` on x86_64).
//! - Zero mutexes, zero spinlocks, safe across Lazy SMP search threads.

use crate::types::Score;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct EvalCacheEntry {
    data: AtomicU64,
}

impl Default for EvalCacheEntry {
    fn default() -> Self {
        Self {
            data: AtomicU64::new(0),
        }
    }
}

pub struct EvalCache {
    entries: Box<[EvalCacheEntry]>,
    mask: usize,
}

impl EvalCache {
    /// Create a new EvalCache with the given number of entries (must be power of two).
    pub fn new(size: usize) -> Self {
        assert!(size.is_power_of_two(), "EvalCache size must be a power of two");
        let mut entries = Vec::with_capacity(size);
        for _ in 0..size {
            entries.push(EvalCacheEntry::default());
        }
        Self {
            entries: entries.into_boxed_slice(),
            mask: size - 1,
        }
    }

    /// Probe the cache for a cached static evaluation.
    /// Returns Some(Score) on hit, None on miss.
    #[inline(always)]
    pub fn probe(&self, hash: u64) -> Option<Score> {
        let idx = (hash as usize) & self.mask;
        let data = self.entries[idx].data.load(Ordering::Relaxed);
        let tag = hash >> 16;
        if data != 0 && (data >> 16) == tag {
            let score_raw = (data as u16) as i16;
            Some(Score::cp(score_raw as i32))
        } else {
            None
        }
    }

    /// Store a static evaluation into the cache.
    #[inline(always)]
    pub fn store(&self, hash: u64, score: Score) {
        let idx = (hash as usize) & self.mask;
        let tag = hash >> 16;
        let packed = (tag << 16) | ((score.raw() as i16 as u16) as u64);
        self.entries[idx].data.store(packed, Ordering::Relaxed);
    }

    /// Clear all cached evaluations.
    pub fn clear(&self) {
        for entry in self.entries.iter() {
            entry.data.store(0, Ordering::Relaxed);
        }
    }
}

impl Default for EvalCache {
    fn default() -> Self {
        // Default to 1M entries (8 MB)
        Self::new(1024 * 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_cache_probe_miss() {
        let cache = EvalCache::new(1024);
        assert_eq!(cache.probe(0x123456789ABCDEF0), None);
    }

    #[test]
    fn test_eval_cache_store_hit() {
        let cache = EvalCache::new(1024);
        let hash = 0xFEDCBA9876543210;
        let score = Score::cp(42);
        cache.store(hash, score);
        assert_eq!(cache.probe(hash), Some(Score::cp(42)));
    }

    #[test]
    fn test_eval_cache_negative_score() {
        let cache = EvalCache::new(1024);
        let hash = 0xAAAAAAAAAAAAAAAA;
        let score = Score::cp(-350);
        cache.store(hash, score);
        assert_eq!(cache.probe(hash), Some(Score::cp(-350)));
    }

    #[test]
    fn test_eval_cache_clear() {
        let cache = EvalCache::new(1024);
        let hash = 0x1111222233334444;
        cache.store(hash, Score::cp(100));
        assert_eq!(cache.probe(hash), Some(Score::cp(100)));
        cache.clear();
        assert_eq!(cache.probe(hash), None);
    }
}
