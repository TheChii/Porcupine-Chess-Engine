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

/// A single entry in the transposition table.
///
/// We use two AtomicU64s to store 128 bits of data.
/// To ensure atomicity without locks, we use the XOR trick:
/// `hash_key = actual_hash ^ data`.
/// When probing, if `hash_key ^ data == actual_hash`, the read is consistent.
pub struct TTBucket {
    hash_key: AtomicU64,
    data: AtomicU64,
}

impl Default for TTBucket {
    fn default() -> Self {
        Self {
            hash_key: AtomicU64::new(0),
            data: AtomicU64::new(0),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TTEntry {
    /// Best move found (encoded)
    best_move: u16,
    /// Evaluation score
    score: i16,
    /// Search depth
    depth: i8,
    /// Bound type (2 bits) + generation/age (6 bits)
    bound_and_age: u8,
}

impl TTEntry {
    /// Create a new TT entry
    pub fn new(
        best_move: Option<Move>,
        score: Score,
        depth: Depth,
        bound: BoundType,
        generation: u8,
    ) -> Self {
        Self {
            best_move: encode_move(best_move),
            score: score.raw() as i16,
            depth: depth.raw() as i8,
            bound_and_age: (bound as u8) | ((generation & 0x3F) << 2),
        }
    }

    /// Pack data into a u64
    /// Layout: padding(16) | best_move(16) | score(16) | depth(8) | bound_and_age(8)
    #[inline]
    pub fn to_data(&self) -> u64 {
        ((self.best_move as u64) << 32)
            | (((self.score as u16) as u64) << 16)
            | ((self.depth as u8 as u64) << 8)
            | (self.bound_and_age as u64)
    }

    /// Unpack data from a u64
    #[inline]
    pub fn from_data(raw: u64) -> Self {
        Self {
            best_move: (raw >> 32) as u16,
            score: (raw >> 16) as i16,
            depth: (raw >> 8) as i8,
            bound_and_age: raw as u8,
        }
    }

    /// Get the bound type
    #[inline]
    pub fn bound(&self) -> BoundType {
        BoundType::from(self.bound_and_age)
    }

    /// Get the generation/age
    #[inline]
    pub fn generation(&self) -> u8 {
        self.bound_and_age >> 2
    }

    /// Get the score
    #[inline]
    pub fn score(&self) -> Score {
        Score::cp(self.score as i32)
    }

    /// Get the depth
    #[inline]
    pub fn depth(&self) -> Depth {
        Depth::new(self.depth as i32)
    }

    /// Get the best move
    #[inline]
    pub fn best_move(&self) -> Option<Move> {
        decode_move(self.best_move)
    }

    /// Check if entry is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bound() == BoundType::None
    }
}

/// Encode a move into 16 bits
/// We can just use the Move's internal bits directly since movegen::Move is already 16 bits
fn encode_move(m: Option<Move>) -> u16 {
    match m {
        Some(mv) => mv.bits(),
        None => 0,
    }
}

/// Decode a 16-bit encoded move
fn decode_move(encoded: u16) -> Option<Move> {
    if encoded == 0 {
        return None;
    }
    Some(Move::from_bits(encoded))
}

/// Lock-free Transposition Table
pub struct TranspositionTable {
    /// Table buckets
    entries: Vec<TTBucket>,
    /// Current generation (incremented each new search)
    generation: AtomicU8,
    /// Size in MB (for reporting)
    size_mb: usize,
}

// Safety: TTBucket only contains Atomics
unsafe impl Send for TranspositionTable {}
unsafe impl Sync for TranspositionTable {}

impl TranspositionTable {
    /// Create a new TT with given size in MB
    pub fn new(size_mb: usize) -> Self {
        // TTBucket is 16 bytes
        let entry_size = 16;
        let num_entries = (size_mb * 1024 * 1024) / entry_size;
        // Round to power of 2 for fast modulo
        let num_entries = num_entries.next_power_of_two() / 2;
        let num_entries = num_entries.max(1024); // Minimum 1024 entries

        let mut entries = Vec::with_capacity(num_entries);
        for _ in 0..num_entries {
            entries.push(TTBucket::default());
        }

        Self {
            entries,
            generation: AtomicU8::new(0),
            size_mb,
        }
    }

    /// Get the number of entries
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if table is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get size in MB
    pub fn size_mb(&self) -> usize {
        self.size_mb
    }

    /// Get current generation
    #[inline]
    pub fn generation(&self) -> u8 {
        self.generation.load(Ordering::Relaxed)
    }

    /// Increment generation (call at start of each search)
    /// Takes &self for thread-safety - uses atomic operation
    pub fn new_search(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    /// Get index for a hash
    #[inline]
    fn index(&self, hash: Hash) -> usize {
        // Fast modulo for power-of-2 size
        (hash as usize) & (self.entries.len() - 1)
    }

    /// Probe the TT for an entry (lock-free)
    #[inline]
    pub fn probe(&self, hash: Hash) -> Option<TTEntry> {
        let bucket = &self.entries[self.index(hash)];
        let data = bucket.data.load(Ordering::Relaxed);
        let hash_key = bucket.hash_key.load(Ordering::Relaxed);

        if (hash_key ^ data) == hash {
            let entry = TTEntry::from_data(data);
            if !entry.is_empty() {
                return Some(entry);
            }
        }
        None
    }

    /// Store an entry in the TT (lock-free)
    ///
    /// Uses depth-preferred replacement with age consideration
    /// Takes &self - uses atomic operations for thread-safety
    pub fn store(
        &self,
        hash: Hash,
        best_move: Option<Move>,
        score: Score,
        depth: Depth,
        bound: BoundType,
    ) {
        let idx = self.index(hash);
        let bucket = &self.entries[idx];
        let existing_data = bucket.data.load(Ordering::Relaxed);
        let existing_key = bucket.hash_key.load(Ordering::Relaxed);
        let gen = self.generation();

        let existing = if (existing_key ^ existing_data) == hash {
            TTEntry::from_data(existing_data)
        } else {
            TTEntry::default() // If hash mismatch, treat as empty (it will be replaced)
        };

        // Replacement strategy:
        // 1. Always replace empty entries or entries from different positions
        // 2. Always replace entries from older generations
        // 3. Replace if new depth >= existing depth
        let should_replace = existing.is_empty()
            || (existing_key ^ existing_data) != hash
            || existing.generation() != gen
            || depth.raw() >= existing.depth.into();

        if should_replace {
            let store_move = best_move.or_else(|| existing.best_move());
            let new_entry = TTEntry::new(store_move, score, depth, bound, gen);
            let new_data = new_entry.to_data();
            let new_key = hash ^ new_data;

            // Write data first, then key
            bucket.data.store(new_data, Ordering::Relaxed);
            bucket.hash_key.store(new_key, Ordering::Relaxed);
        }
    }

    /// Clear the table
    pub fn clear(&self) {
        for bucket in &self.entries {
            bucket.data.store(0, Ordering::Relaxed);
            bucket.hash_key.store(0, Ordering::Relaxed);
        }
        self.generation.store(0, Ordering::Relaxed);
    }

    /// Get hashfull in permill (for UCI info)
    pub fn hashfull(&self) -> u32 {
        let gen = self.generation();
        // Sample first 1000 entries
        let sample_size = self.entries.len().min(1000);
        let used = self.entries[..sample_size]
            .iter()
            .filter(|b| {
                let data = b.data.load(Ordering::Relaxed);
                let entry = TTEntry::from_data(data);
                !entry.is_empty() && entry.generation() == gen
            })
            .count();
        ((used * 1000) / sample_size) as u32
    }

    /// Prefetch entry for a hash (performance optimization)
    /// Uses CPU prefetch intrinsics to bring TT entry into L1 cache
    #[inline]
    pub fn prefetch(&self, hash: Hash) {
        let idx = self.index(hash);
        let ptr = self.entries.as_ptr().wrapping_add(idx) as *const i8;

        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr, std::arch::x86_64::_MM_HINT_T0);
        }

        #[cfg(target_arch = "x86")]
        unsafe {
            std::arch::x86::_mm_prefetch(ptr, std::arch::x86::_MM_HINT_T0);
        }

        // No-op on other architectures
        #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
        let _ = ptr;
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
