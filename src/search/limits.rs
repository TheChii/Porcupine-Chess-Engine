//! Search limits and time management.
//!
//! Handles:
//! - Fixed depth search
//! - Fixed time search
//! - Time control with increment
//! - Infinite search (until stop)
//! - Soft/hard time limits for optimal iteration control

use crate::types::{Color, Depth};
use crate::uci::SearchParams;
use std::time::Instant;

/// Search limits configuration
#[derive(Debug, Clone, Default)]
pub struct SearchLimits {
    /// Maximum depth to search
    pub depth: Option<Depth>,
    /// Maximum time in milliseconds
    pub movetime: Option<u64>,
    /// Maximum nodes to search
    pub nodes: Option<u64>,
    /// White time remaining (ms)
    pub wtime: Option<u64>,
    /// Black time remaining (ms)
    pub btime: Option<u64>,
    /// White increment (ms)
    pub winc: Option<u64>,
    /// Black increment (ms)
    pub binc: Option<u64>,
    /// Moves until next time control
    pub movestogo: Option<u32>,
    /// Infinite search
    pub infinite: bool,
    /// Ponder mode
    pub ponder: bool,
    /// Move overhead (safety buffer for network/GUI delay)
    pub move_overhead: u64,
}

impl SearchLimits {
    pub const DEFAULT_MOVE_OVERHEAD: u64 = 50;

    pub fn new() -> Self { Self { move_overhead: Self::DEFAULT_MOVE_OVERHEAD, ..Default::default() } }

    pub fn depth(d: i32) -> Self { Self { depth: Some(Depth::new(d)), move_overhead: Self::DEFAULT_MOVE_OVERHEAD, ..Default::default() } }

    pub fn from_params(p: &SearchParams) -> Self {
        Self {
            depth: p.depth, movetime: p.movetime, nodes: p.nodes,
            wtime: p.wtime, btime: p.btime, winc: p.winc, binc: p.binc,
            movestogo: p.movestogo, infinite: p.infinite, ponder: p.ponder,
            move_overhead: Self::DEFAULT_MOVE_OVERHEAD,
        }
    }

    pub fn with_move_overhead(mut self, o: u64) -> Self { self.move_overhead = o; self }
}

/// Time manager for search with soft and hard limits
#[derive(Debug, Clone)]
pub struct TimeManager {
    /// Soft time limit - target time to use (stop after iteration)
    soft_limit: u64,
    /// Hard time limit - absolute maximum (stop mid-search if exceeded)
    hard_limit: u64,
    /// Move overhead safety buffer
    _move_overhead: u64,
    /// Is this an infinite search?
    infinite: bool,
    /// Are we currently pondering?
    ponder: bool,
    /// Start time of search
    start_time: Option<Instant>,
}

impl TimeManager {
    pub fn new() -> Self {
        Self { soft_limit: u64::MAX, hard_limit: u64::MAX, _move_overhead: 10, infinite: true, ponder: false, start_time: Some(Instant::now()) }
    }

    /// Create time manager from search limits
    pub fn from_limits(limits: &SearchLimits, side: Color, moves_played: u16) -> Self {
        if limits.infinite {
            return Self::new();
        }

        let move_overhead = limits.move_overhead;

        // Fixed movetime - use more time since we have a hard budget
        // Soft limit: 92% of available time (when to consider stopping after iteration)
        // Hard limit: 98% of available time (absolute stop, leave small buffer)
        if let Some(mt) = limits.movetime {
            let available = mt.saturating_sub(move_overhead);
            // Use 92% for soft limit - try to complete more iterations
            let soft = (available * 92) / 100;
            // Use 98% for hard limit - leave only 2% buffer for move transmission
            let hard = (available * 98) / 100;
            return Self {
                soft_limit: soft.max(1),
                hard_limit: hard.max(1),
                _move_overhead: move_overhead,
                infinite: false,
                ponder: limits.ponder,
                start_time: Some(Instant::now()),
            };
        }

        // Time control with wtime/btime
        let (time_left, increment) = match side {
            Color::White => (limits.wtime, limits.winc),
            Color::Black => (limits.btime, limits.binc),
        };

        if let Some(time) = time_left {
            let inc = increment.unwrap_or(0);

            // Subtract overhead from available time
            let available = time.saturating_sub(move_overhead);

            // Estimate moves remaining based on time situation
            let mtg = if let Some(movestogo) = limits.movestogo {
                // Explicit moves to go (sudden death with X moves per period)
                // Cap at 30 to spend more time early in the period
                (movestogo as u64).min(30)
            } else {
                // No explicit moves-to-go, estimate based on time left
                // Use a more conservative estimate for blitz to avoid burning time
                if available > 300000 {
                    40
                } else if available > 60000 {
                    35
                } else if available > 10000 {
                    30
                } else {
                    20 // Don't drop below 20 moves to go to prevent burning base time
                }
            }
            .max(1);

            // Base time allocation per move
            let mut base_time = available / mtg;

            // Reduce time in the opening to save for midgame
            if moves_played <= 10 {
                let factor = 50 + (moves_played as u64 * 5); // 55% at move 1, up to 100% at move 10
                base_time = (base_time * factor) / 100;
            }

            // Add most of increment to our budget (we'll get it back after moving)
            let inc_bonus = (inc * 85) / 100; // Use 85% of increment

            // Soft limit: base + increment bonus, but cap at reasonable portion of remaining time
            let max_soft = if mtg <= 1 { available * 80 / 100 } else if mtg <= 3 { available * 40 / 100 } else { available / 3 };
            let soft = (base_time + inc_bonus).min(max_soft);

            // Hard limit: allow up to 4x soft for critical moves, but never more than max_hard
            let max_hard = if mtg <= 1 { available * 95 / 100 } else if mtg <= 3 { available * 60 / 100 } else { available / 2 };
            let hard = (soft * 4).min(max_hard).max(soft);

            // Minimum thresholds to avoid instant moves, but NEVER exceed available time!
            let soft = soft.max(50).min(available); // At least 50ms, or all available
            let hard = hard.max(100).min(available); // At least 100ms, or all available

            return Self {
                soft_limit: soft,
                hard_limit: hard,
                _move_overhead: move_overhead,
                infinite: false,
                ponder: limits.ponder,
                start_time: Some(Instant::now()),
            };
        }

        // Fallback to infinite (but with timer started)
        Self {
            soft_limit: u64::MAX,
            hard_limit: u64::MAX,
            _move_overhead: move_overhead,
            infinite: true,
            ponder: limits.ponder,
            start_time: Some(Instant::now()),
        }
    }

    pub fn start(&mut self) { self.start_time = Some(Instant::now()); }
    pub fn elapsed(&self) -> u64 { self.start_time.map(|t| t.elapsed().as_millis() as u64).unwrap_or(0) }
    pub fn should_stop(&self) -> bool { if self.infinite || self.ponder { false } else { self.elapsed() >= self.hard_limit } }
    pub fn can_start_iteration(&self) -> bool { if self.infinite || self.ponder { true } else { self.elapsed() < self.soft_limit } }
    pub fn soft_limit_exceeded(&self) -> bool { if self.infinite || self.ponder { false } else { self.elapsed() >= self.soft_limit } }
    pub fn hard_limit_exceeded(&self) -> bool { if self.infinite || self.ponder { false } else { self.elapsed() >= self.hard_limit } }

    #[allow(dead_code)]
    pub fn extend_time(&mut self, factor: f64) {
        if !self.infinite {
            self.soft_limit = ((self.soft_limit as f64) * factor) as u64;
            self.hard_limit = ((self.hard_limit as f64) * factor.sqrt()) as u64;
        }
    }

    pub fn soft_limit_ms(&self) -> u64 { self.soft_limit }
    pub fn hard_limit_ms(&self) -> u64 { self.hard_limit }
    pub fn is_infinite(&self) -> bool { self.infinite || self.ponder }
    pub fn ponderhit(&mut self) { self.ponder = false; }
}

impl Default for TimeManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_movetime() {
        let limits = SearchLimits {
            movetime: Some(1000),
            move_overhead: 50,
            ..Default::default()
        };
        let tm = TimeManager::from_limits(&limits, Color::White, 1);

        assert!(!tm.is_infinite());
        // 1000 - 50 overhead = 950 available
        // soft = 950 * 92% = 874
        // hard = 950 * 98% = 931
        assert_eq!(tm.soft_limit_ms(), 874);
        assert_eq!(tm.hard_limit_ms(), 931);
    }

    #[test]
    fn test_time_control() {
        let limits = SearchLimits {
            wtime: Some(60000),
            btime: Some(60000),
            winc: Some(1000),
            binc: Some(1000),
            move_overhead: 10,
            ..Default::default()
        };
        let tm = TimeManager::from_limits(&limits, Color::White, 1);

        assert!(!tm.is_infinite());
        // 60000 - 10 = 59990 available
        // mtg = 30 (since available <= 60000)
        // base = 59990 / 30 = 1999
        // Opening reduction: move 1 = 55% -> 1099
        // inc_bonus = 1000 * 0.85 = 850
        // soft = 1949
        assert!(tm.soft_limit_ms() > 1900);
        assert!(tm.soft_limit_ms() < 2000);
        // hard = min(3 * soft, available / 4)
        assert!(tm.hard_limit_ms() >= tm.soft_limit_ms());
    }

    #[test]
    fn test_infinite() {
        let limits = SearchLimits {
            infinite: true,
            ..Default::default()
        };
        let tm = TimeManager::from_limits(&limits, Color::White, 1);

        assert!(tm.is_infinite());
        assert!(tm.can_start_iteration());
        assert!(!tm.should_stop());
    }
}
