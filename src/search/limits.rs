//! Search limits and time management.
//!
//! Handles:
//! - Fixed depth search
//! - Fixed time search
//! - Time control with increment
//! - Infinite search (until stop)
//! - Soft/hard time limits for optimal iteration control

use crate::types::{Depth, Color};
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
    /// Move overhead (safety buffer for network/GUI delay)
    pub move_overhead: u64,
}

impl SearchLimits {
    /// Default move overhead for timing safety (ms)
    pub const DEFAULT_MOVE_OVERHEAD: u64 = 50;
    
    pub fn new() -> Self {
        Self {
            move_overhead: Self::DEFAULT_MOVE_OVERHEAD,
            ..Default::default()
        }
    }

    pub fn depth(depth: i32) -> Self {
        Self {
            depth: Some(Depth::new(depth)),
            move_overhead: Self::DEFAULT_MOVE_OVERHEAD,
            ..Default::default()
        }
    }

    pub fn from_params(params: &SearchParams) -> Self {
        Self {
            depth: params.depth,
            movetime: params.movetime,
            nodes: params.nodes,
            wtime: params.wtime,
            btime: params.btime,
            winc: params.winc,
            binc: params.binc,
            movestogo: params.movestogo,
            infinite: params.infinite,
            move_overhead: Self::DEFAULT_MOVE_OVERHEAD,
        }
    }
    
    /// Set move overhead (from UCI option)
    pub fn with_move_overhead(mut self, overhead: u64) -> Self {
        self.move_overhead = overhead;
        self
    }
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
    /// Start time of search
    start_time: Option<Instant>,
}

impl TimeManager {
    pub fn new() -> Self {
        Self {
            soft_limit: u64::MAX,
            hard_limit: u64::MAX,
            _move_overhead: 10,
            infinite: true,
            start_time: Some(Instant::now()),
        }
    }

    /// Create time manager from search limits
    pub fn from_limits(limits: &SearchLimits, side: Color) -> Self {
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
                movestogo as u64
            } else {
                // No explicit moves-to-go, estimate based on time left
                // Use a dynamic estimate: more time = more conservative
                // Less time = more aggressive (assume game is ending)
                if available > 300000 {
                    // > 5 min: assume 40 moves remaining
                    40
                } else if available > 120000 {
                    // 2-5 min: assume 30 moves 
                    30
                } else if available > 60000 {
                    // 1-2 min: assume 25 moves
                    25
                } else if available > 30000 {
                    // 30s-1min: assume 20 moves
                    20
                } else if available > 10000 {
                    // 10-30s: assume 15 moves
                    15
                } else {
                    // < 10s: panic mode, 10 moves
                    10
                }
            }.max(1);
            
            // Base time allocation per move
            let base_time = available / mtg;
            
            // Add most of increment to our budget (we'll get it back after moving)
            let inc_bonus = (inc * 85) / 100;  // Use 85% of increment
            
            // Soft limit: base + increment bonus, but cap at reasonable portion of remaining time
            let soft = (base_time + inc_bonus).min(available / 3);
            
            // Hard limit: allow up to 3x soft for critical moves, but never more than 50% of remaining
            let hard = (soft * 3).min(available / 2).max(soft);
            
            // Minimum thresholds to avoid instant moves
            let soft = soft.max(100); // At least 100ms
            let hard = hard.max(200); // At least 200ms
            
            return Self {
                soft_limit: soft,
                hard_limit: hard,
                _move_overhead: move_overhead,
                infinite: false,
                start_time: Some(Instant::now()),
            };
        }

        // Fallback to infinite (but with timer started)
        Self {
            soft_limit: u64::MAX,
            hard_limit: u64::MAX,
            _move_overhead: move_overhead,
            infinite: true,
            start_time: Some(Instant::now()),
        }
    }
    
    /// Start the timer (call at search start)
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed(&self) -> u64 {
        self.start_time
            .map(|t| t.elapsed().as_millis() as u64)
            .unwrap_or(0)
    }

    /// Check if we should stop searching (hard limit - for mid-search check)
    pub fn should_stop(&self) -> bool {
        if self.infinite {
            return false;
        }
        self.elapsed() >= self.hard_limit
    }

    /// Check if we can start a new iteration (soft limit)
    pub fn can_start_iteration(&self) -> bool {
        if self.infinite {
            return true;
        }
        // Start new iteration if we have time remaining below soft limit
        // and predict we can complete at least a partial iteration
        self.elapsed() < self.soft_limit
    }

    /// Check if we've exceeded soft limit (use between iterations)
    pub fn soft_limit_exceeded(&self) -> bool {
        if self.infinite {
            return false;
        }
        self.elapsed() >= self.soft_limit
    }

    /// Hard stop check (absolute limit - never exceed)
    pub fn hard_limit_exceeded(&self) -> bool {
        if self.infinite {
            return false;
        }
        self.elapsed() >= self.hard_limit
    }
    
    /// Extend time limits (when search is in trouble, e.g., score dropped)
    /// factor > 1.0 extends time, factor < 1.0 reduces time
    #[allow(dead_code)]
    pub fn extend_time(&mut self, factor: f64) {
        if !self.infinite {
            self.soft_limit = ((self.soft_limit as f64) * factor) as u64;
            // Hard limit extends less aggressively
            self.hard_limit = ((self.hard_limit as f64) * factor.sqrt()) as u64;
        }
    }
    
    /// Get the soft limit in ms
    pub fn soft_limit_ms(&self) -> u64 {
        self.soft_limit
    }
    
    /// Get the hard limit in ms
    pub fn hard_limit_ms(&self) -> u64 {
        self.hard_limit
    }
    
    /// Check if this is an infinite search
    pub fn is_infinite(&self) -> bool {
        self.infinite
    }
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
        let tm = TimeManager::from_limits(&limits, Color::White);
        
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
        let tm = TimeManager::from_limits(&limits, Color::White);
        
        assert!(!tm.is_infinite());
        // 60000 - 10 = 59990 available
        // base = 59990 / 30 = ~1999
        // inc_bonus = 1000 * 0.75 = 750
        // soft = ~2749
        assert!(tm.soft_limit_ms() > 2000);
        assert!(tm.soft_limit_ms() < 4000);
        // hard = min(3 * soft, available / 4)
        assert!(tm.hard_limit_ms() >= tm.soft_limit_ms());
    }
    
    #[test]
    fn test_infinite() {
        let limits = SearchLimits {
            infinite: true,
            ..Default::default()
        };
        let tm = TimeManager::from_limits(&limits, Color::White);
        
        assert!(tm.is_infinite());
        assert!(tm.can_start_iteration());
        assert!(!tm.should_stop());
    }
}
