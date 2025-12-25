//! Transposition Table for caching search results.
//!
//! This module provides a high-performance, lock-free transposition table
//! that stores search results to avoid redundant computation.
//!
//! # Design
//! - 16-byte entries for cache efficiency
//! - Depth-preferred replacement with age-based eviction
//! - Lock-free for future multi-threading support

use crate::types::{Move, Score, Depth, Hash};

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
/// Packed into 16 bytes for cache efficiency:
/// - key: 2 bytes (upper bits of hash for verification)
/// - best_move: 2 bytes (encoded move)
/// - score: 2 bytes
/// - depth: 1 byte
/// - bound_and_age: 1 byte (bound type in low 2 bits, age in high 6 bits)
/// - padding: 8 bytes (for alignment, could store more data)
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct TTEntry {
    /// Upper 16 bits of Zobrist hash for verification
    key: u16,
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
        hash: Hash,
        best_move: Option<Move>,
        score: Score,
        depth: Depth,
        bound: BoundType,
        generation: u8,
    ) -> Self {
        Self {
            key: (hash >> 48) as u16,
            best_move: encode_move(best_move),
            score: score.raw() as i16,
            depth: depth.raw() as i8,
            bound_and_age: (bound as u8) | ((generation & 0x3F) << 2),
        }
    }

    /// Check if entry matches the given hash
    #[inline]
    pub fn matches(&self, hash: Hash) -> bool {
        self.key == (hash >> 48) as u16
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

/// Encode a move into 16 bits: from (6) + to (6) + promo (4)
fn encode_move(m: Option<Move>) -> u16 {
    match m {
        Some(mv) => {
            let from = mv.get_source().to_index() as u16;
            let to = mv.get_dest().to_index() as u16;
            let promo = match mv.get_promotion() {
                Some(chess::Piece::Knight) => 1,
                Some(chess::Piece::Bishop) => 2,
                Some(chess::Piece::Rook) => 3,
                Some(chess::Piece::Queen) => 4,
                _ => 0,
            };
            (from) | (to << 6) | (promo << 12)
        }
        None => 0,
    }
}

/// Decode a 16-bit encoded move
fn decode_move(encoded: u16) -> Option<Move> {
    if encoded == 0 {
        return None;
    }

    let from_idx = (encoded & 0x3F) as u8;
    let to_idx = ((encoded >> 6) & 0x3F) as u8;
    let promo_bits = (encoded >> 12) & 0x0F;

    // Square::new is unsafe because it doesn't validate the index
    // We know our indices are valid (0-63) from the encoding
    let from = unsafe { chess::Square::new(from_idx) };
    let to = unsafe { chess::Square::new(to_idx) };

    let promo = match promo_bits {
        1 => Some(chess::Piece::Knight),
        2 => Some(chess::Piece::Bishop),
        3 => Some(chess::Piece::Rook),
        4 => Some(chess::Piece::Queen),
        _ => None,
    };

    Some(Move::new(from, to, promo))
}

/// The Transposition Table
pub struct TranspositionTable {
    /// Table entries
    entries: Vec<TTEntry>,
    /// Current generation (incremented each new search)
    generation: u8,
    /// Size in MB (for reporting)
    size_mb: usize,
}

impl TranspositionTable {
    /// Create a new TT with given size in MB
    pub fn new(size_mb: usize) -> Self {
        let entry_size = std::mem::size_of::<TTEntry>();
        let num_entries = (size_mb * 1024 * 1024) / entry_size;
        // Round to power of 2 for fast modulo
        let num_entries = num_entries.next_power_of_two() / 2;
        let num_entries = num_entries.max(1024); // Minimum 1024 entries

        Self {
            entries: vec![TTEntry::default(); num_entries],
            generation: 0,
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

    /// Increment generation (call at start of each search)
    pub fn new_search(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    /// Get index for a hash
    #[inline]
    fn index(&self, hash: Hash) -> usize {
        // Fast modulo for power-of-2 size
        (hash as usize) & (self.entries.len() - 1)
    }

    /// Probe the TT for an entry
    #[inline]
    pub fn probe(&self, hash: Hash) -> Option<&TTEntry> {
        let entry = &self.entries[self.index(hash)];
        if entry.matches(hash) && !entry.is_empty() {
            Some(entry)
        } else {
            None
        }
    }

    /// Store an entry in the TT
    ///
    /// Uses depth-preferred replacement with age consideration
    pub fn store(
        &mut self,
        hash: Hash,
        best_move: Option<Move>,
        score: Score,
        depth: Depth,
        bound: BoundType,
    ) {
        let idx = self.index(hash);
        let existing = &self.entries[idx];

        // Replacement strategy:
        // 1. Always replace empty entries
        // 2. Always replace entries from older generations
        // 3. Replace if new depth >= existing depth
        let should_replace = existing.is_empty()
            || existing.generation() != self.generation
            || depth.raw() >= existing.depth.into();

        if should_replace {
            self.entries[idx] = TTEntry::new(hash, best_move, score, depth, bound, self.generation);
        }
    }

    /// Clear the table
    pub fn clear(&mut self) {
        for entry in &mut self.entries {
            *entry = TTEntry::default();
        }
        self.generation = 0;
    }

    /// Resize the table (typically from UCI setoption)
    pub fn resize(&mut self, size_mb: usize) {
        *self = TranspositionTable::new(size_mb);
    }

    /// Get hashfull in permill (for UCI info)
    pub fn hashfull(&self) -> u32 {
        // Sample first 1000 entries
        let sample_size = self.entries.len().min(1000);
        let used = self.entries[..sample_size]
            .iter()
            .filter(|e| !e.is_empty() && e.generation() == self.generation)
            .count();
        ((used * 1000) / sample_size) as u32
    }

    /// Prefetch entry for a hash (performance optimization)
    /// Currently a no-op, can be enabled with target-specific intrinsics
    #[inline]
    pub fn prefetch(&self, hash: Hash) {
        // Compute index to potentially trigger cache-friendly access
        let _ = self.index(hash);
        // Future: use platform-specific prefetch intrinsics
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

    #[test]
    fn test_tt_basic() {
        let mut tt = TranspositionTable::new(1);
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
        let mv = Move::new(
            chess::Square::E2,
            chess::Square::E4,
            None,
        );
        let encoded = encode_move(Some(mv));
        let decoded = decode_move(encoded).unwrap();
        assert_eq!(mv.get_source(), decoded.get_source());
        assert_eq!(mv.get_dest(), decoded.get_dest());
    }
}
