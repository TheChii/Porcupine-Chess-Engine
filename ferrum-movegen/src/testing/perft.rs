//! Perft (performance test) function for move generator validation.

use crate::board::Board;

/// Run perft to a given depth.
/// Returns the number of leaf nodes at the given depth.
pub fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    if depth == 1 {
        return board.generate_moves_count();
    }

    let moves = board.generate_moves();

    let mut nodes = 0u64;

    for mv in &moves {
        let new_board = board.make_move_new(mv);
        nodes += perft(&new_board, depth - 1);
    }

    nodes
}

/// Run perft with divide (shows count per root move).
pub fn perft_divide(board: &Board, depth: u32) -> Vec<(String, u64)> {
    let moves = board.generate_moves();
    let mut results = Vec::new();

    for mv in &moves {
        let new_board = board.make_move_new(mv);
        let count = perft(&new_board, depth - 1);
        results.push((mv.to_uci(), count));
    }

    results
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perft_startpos_depth1() {
        let board = Board::startpos();
        assert_eq!(perft(&board, 1), 20);
    }

    #[test]
    fn test_perft_startpos_depth2() {
        let board = Board::startpos();
        assert_eq!(perft(&board, 2), 400);
    }

    #[test]
    fn test_perft_startpos_depth3() {
        let board = Board::startpos();
        assert_eq!(perft(&board, 3), 8902);
    }

    #[test]
    fn test_perft_startpos_depth4() {
        let board = Board::startpos();
        assert_eq!(perft(&board, 4), 197281);
    }

    #[test]
    fn test_debug_perft_divide() {
        let board = Board::startpos();
        let results = perft_divide(&board, 4);
        let total: u64 = results.iter().map(|(_, c)| c).sum();
        println!("Total: {} (expected 197281)", total);
        for (mv, count) in &results {
            println!("{}: {}", mv, count);
        }
        // Known correct values for startpos depth 4:
        // a2a3: 8457, a2a4: 9329, b1a3: 8885, b1c3: 9755
        // b2b3: 9345, b2b4: 9332, c2c3: 9272, c2c4: 9744
        // d2d3: 11959, d2d4: 12435, e2e3: 13134, e2e4: 13160
        // f2f3: 8457, f2f4: 8929, g1f3: 9748, g1h3: 8881
        // g2g3: 9345, g2g4: 9328, h2h3: 8457, h2h4: 9329
    }

    #[test]
    #[ignore] // Slow test
    fn test_perft_startpos_depth5() {
        let board = Board::startpos();
        assert_eq!(perft(&board, 5), 4865609);
    }

    #[test]
    fn test_perft_kiwipete_depth1() {
        let board = Board::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        ).unwrap();
        assert_eq!(perft(&board, 1), 48);
    }

    #[test]
    fn test_perft_kiwipete_depth2() {
        let board = Board::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        ).unwrap();
        assert_eq!(perft(&board, 2), 2039);
    }
}
