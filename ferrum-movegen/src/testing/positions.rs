//! Standard test positions for perft validation.

/// Starting position.
pub const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// Kiwipete - a complex position with many tactical elements.
pub const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

/// Position 3 - tests en passant and promotion.
pub const POSITION_3: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";

/// Position 4 - complex middlegame.
pub const POSITION_4: &str = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";

/// Position 5 - another complex position.
pub const POSITION_5: &str = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";

/// Position 6.
pub const POSITION_6: &str = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";

/// Standard perft results for verification.
pub struct PerftResult {
    pub fen: &'static str,
    pub depth: u32,
    pub nodes: u64,
}

/// Known perft results for validation.
pub const PERFT_RESULTS: &[PerftResult] = &[
    // Starting position
    PerftResult { fen: STARTPOS, depth: 1, nodes: 20 },
    PerftResult { fen: STARTPOS, depth: 2, nodes: 400 },
    PerftResult { fen: STARTPOS, depth: 3, nodes: 8902 },
    PerftResult { fen: STARTPOS, depth: 4, nodes: 197281 },
    PerftResult { fen: STARTPOS, depth: 5, nodes: 4865609 },
    
    // Kiwipete
    PerftResult { fen: KIWIPETE, depth: 1, nodes: 48 },
    PerftResult { fen: KIWIPETE, depth: 2, nodes: 2039 },
    PerftResult { fen: KIWIPETE, depth: 3, nodes: 97862 },
    PerftResult { fen: KIWIPETE, depth: 4, nodes: 4085603 },
    
    // Position 3
    PerftResult { fen: POSITION_3, depth: 1, nodes: 14 },
    PerftResult { fen: POSITION_3, depth: 2, nodes: 191 },
    PerftResult { fen: POSITION_3, depth: 3, nodes: 2812 },
    PerftResult { fen: POSITION_3, depth: 4, nodes: 43238 },
    PerftResult { fen: POSITION_3, depth: 5, nodes: 674624 },
    
    // Position 4
    PerftResult { fen: POSITION_4, depth: 1, nodes: 6 },
    PerftResult { fen: POSITION_4, depth: 2, nodes: 264 },
    PerftResult { fen: POSITION_4, depth: 3, nodes: 9467 },
    PerftResult { fen: POSITION_4, depth: 4, nodes: 422333 },
    
    // Position 5
    PerftResult { fen: POSITION_5, depth: 1, nodes: 44 },
    PerftResult { fen: POSITION_5, depth: 2, nodes: 1486 },
    PerftResult { fen: POSITION_5, depth: 3, nodes: 62379 },
    PerftResult { fen: POSITION_5, depth: 4, nodes: 2103487 },
];
