//! Benchmark suite for FH-NNUE evaluation and search performance.
//!
//! Measures:
//! 1. Model load time
//! 2. Memory footprint
//! 3. Full accumulator refresh latency
//! 4. Incremental accumulator update latency
//! 5. Dense head inference latency
//! 6. Complete NNUE evaluation throughput
//! 7. Search nodes per second (HCE vs FH-NNUE)

use porcupine::nnue::{
    evaluate_from_accumulators, load_global_model, set_use_nnue, NnueModel,
    PerspectiveAccumulators,
};
use porcupine::search::{SearchLimits, Searcher};
use porcupine::types::{Board, Depth};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    println!("============================================================");
    println!("             FH-NNUE Performance Benchmark Suite            ");
    println!("============================================================");

    // 1. Model Load Time & Memory Footprint
    println!("\n--- 1. Model Loading and Memory Footprint ---");
    let model_path = "default.nnue";
    let t_load_start = Instant::now();
    let model = NnueModel::load_from_file(model_path).expect("Failed to load default.nnue");
    let load_duration = t_load_start.elapsed();

    let ft_bytes = model.feature_weights.len() * 512 * 2;
    let ft_bias_bytes = 512 * 2;
    let psqt_bytes = model.psqt_weights.as_ref().map(|w| w.len() * 8 * 4).unwrap_or(0);
    let head_bytes = std::mem::size_of_val(&model.head);
    let total_bytes = ft_bytes + ft_bias_bytes + psqt_bytes + head_bytes;

    println!("Model loaded from: {}", model_path);
    println!("Model load time:   {:.3} ms", load_duration.as_secs_f64() * 1000.0);
    println!("Feature weights:   {:.2} MB", ft_bytes as f64 / (1024.0 * 1024.0));
    println!("PSQT weights:      {:.2} MB", psqt_bytes as f64 / (1024.0 * 1024.0));
    println!("Dense head:        {:.2} KB", head_bytes as f64 / 1024.0);
    println!("Total model heap:  {:.2} MB", total_bytes as f64 / (1024.0 * 1024.0));

    // Test positions
    let positions = [
        ("Startpos", Board::startpos()),
        (
            "Middlegame (Ruy Lopez)",
            Board::from_fen("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3").unwrap(),
        ),
        (
            "Tactical (Queen capture)",
            Board::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap(),
        ),
        (
            "Endgame (K+R vs K)",
            Board::from_fen("8/8/8/4k3/8/8/4K3/4R3 w - - 0 1").unwrap(),
        ),
        (
            "Castling rights complex",
            Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap(),
        ),
    ];

    let psqt = model.psqt_weights.as_deref();

    // 2. Full Accumulator Refresh Latency
    println!("\n--- 2. Full Accumulator Refresh Latency ---");
    const REFRESH_ITERS: usize = 50_000;
    for (name, board) in &positions {
        let mut refresh_sink = 0i32;
        let t0 = Instant::now();
        for _ in 0..REFRESH_ITERS {
            let acc = PerspectiveAccumulators::refresh(
                board,
                &model.feature_biases,
                &model.feature_weights,
                &model.psqt_biases,
                psqt,
            );
            refresh_sink = refresh_sink.wrapping_add(acc.white.vals[0] as i32);
        }
        black_box(refresh_sink);
        let elapsed = t0.elapsed();
        let ns_per_op = elapsed.as_nanos() as f64 / REFRESH_ITERS as f64;
        let ops_per_sec = REFRESH_ITERS as f64 / elapsed.as_secs_f64();
        println!(
            "{:<24} | {:>7.1} ns/op | {:>9.0} ops/sec",
            name, ns_per_op, ops_per_sec
        );
    }

    // 3. Incremental Accumulator Update Latency
    println!("\n--- 3. Incremental Accumulator Update Latency ---");
    const UPDATE_ITERS: usize = 200_000;
    for (name, board) in &positions {
        let root_acc = PerspectiveAccumulators::refresh(
            board,
            &model.feature_biases,
            &model.feature_weights,
            &model.psqt_biases,
            psqt,
        );
        let moves = board.generate_moves();
        if moves.is_empty() {
            continue;
        }
        let mv = moves.as_slice()[0];
        let new_board = board.make_move_new(mv);

        let mut update_sink = 0i32;
        let t0 = Instant::now();
        for _ in 0..UPDATE_ITERS {
            let next_acc = root_acc.update_move(
                board,
                &new_board,
                mv,
                &model.feature_biases,
                &model.feature_weights,
                &model.psqt_biases,
                psqt,
            );
            update_sink = update_sink.wrapping_add(next_acc.white.vals[0] as i32);
        }
        black_box(update_sink);
        let elapsed = t0.elapsed();
        let ns_per_op = elapsed.as_nanos() as f64 / UPDATE_ITERS as f64;
        let ops_per_sec = UPDATE_ITERS as f64 / elapsed.as_secs_f64();
        println!(
            "{:<24} | {:>7.1} ns/op | {:>9.0} ops/sec",
            name, ns_per_op, ops_per_sec
        );
    }

    // 4. Dense Head Inference Latency
    println!("\n--- 4. Quantized Dense Head Inference Latency ---");
    const DENSE_ITERS: usize = 200_000;
    let root_acc = PerspectiveAccumulators::refresh(
        &positions[0].1,
        &model.feature_biases,
        &model.feature_weights,
        &model.psqt_biases,
        psqt,
    );
    let mut acc_w = root_acc.white.clone();
    let acc_b = root_acc.black.clone();
    let mut dense_sink = 0i32;
    let t0 = Instant::now();
    for i in 0..DENSE_ITERS {
        // Toggle 1 bit in accumulator to prevent loop invariant code motion/hoisting
        acc_w.vals[i & 511] ^= 1;
        let score = model.head.evaluate(black_box(&acc_w), black_box(&acc_b));
        dense_sink = dense_sink.wrapping_add(score.raw());
    }
    black_box(dense_sink);
    let elapsed = t0.elapsed();
    let ns_per_op = elapsed.as_nanos() as f64 / DENSE_ITERS as f64;
    let ops_per_sec = DENSE_ITERS as f64 / elapsed.as_secs_f64();
    println!("Dense head (1024->128->1) | {:>7.1} ns/op | {:>9.0} evaluations/sec", ns_per_op, ops_per_sec);

    // 5. Complete Incremental Evaluation Latency
    println!("\n--- 5. Complete Incremental Evaluation Latency ---");
    let moves = positions[0].1.generate_moves();
    let mv = moves.as_slice()[0];
    let new_b = positions[0].1.make_move_new(mv);
    let mut eval_sink = 0i32;
    let t0 = Instant::now();
    for _ in 0..UPDATE_ITERS {
        let next_acc = root_acc.update_move(
            &positions[0].1,
            &new_b,
            mv,
            &model.feature_biases,
            &model.feature_weights,
            &model.psqt_biases,
            psqt,
        );
        let score = evaluate_from_accumulators(&next_acc, new_b.turn(), &model);
        eval_sink = eval_sink.wrapping_add(score.raw());
    }
    black_box(eval_sink);
    let elapsed = t0.elapsed();
    let ns_per_op = elapsed.as_nanos() as f64 / UPDATE_ITERS as f64;
    let ops_per_sec = UPDATE_ITERS as f64 / elapsed.as_secs_f64();
    println!("Incremental Eval (Update + Head) | {:>7.1} ns/op | {:>9.0} evals/sec", ns_per_op, ops_per_sec);

    // 6. Search NPS Comparison: HCE vs FH-NNUE
    println!("\n--- 6. Search NPS Comparison (Depth 9 Search) ---");
    load_global_model(model_path).unwrap();

    let search_pos = Board::startpos();
    let depth = Depth::new(9);

    // Run HCE search
    set_use_nnue(false);
    let mut searcher_hce = Searcher::new();
    searcher_hce.set_position(search_pos);
    let mut limits_hce = SearchLimits::default();
    limits_hce.depth = Some(depth);
    let t_hce = Instant::now();
    let res_hce = searcher_hce.search(limits_hce);
    let elapsed_hce = t_hce.elapsed();
    let nodes_hce = searcher_hce.stats().nodes;
    let nps_hce = if elapsed_hce.as_secs_f64() > 0.0 {
        nodes_hce as f64 / elapsed_hce.as_secs_f64()
    } else {
        0.0
    };

    println!(
        "HCE     | Best: {:?} | Score: {:?} | Nodes: {:>7} | Time: {:>5.1} ms | NPS: {:>9.0}",
        res_hce.best_move,
        res_hce.score,
        nodes_hce,
        elapsed_hce.as_secs_f64() * 1000.0,
        nps_hce
    );

    // Run NNUE search
    set_use_nnue(true);
    let mut searcher_nnue = Searcher::new();
    searcher_nnue.set_position(search_pos);
    let mut limits_nnue = SearchLimits::default();
    limits_nnue.depth = Some(depth);
    let t_nnue = Instant::now();
    let res_nnue = searcher_nnue.search(limits_nnue);
    let elapsed_nnue = t_nnue.elapsed();
    let nodes_nnue = searcher_nnue.stats().nodes;
    let nps_nnue = if elapsed_nnue.as_secs_f64() > 0.0 {
        nodes_nnue as f64 / elapsed_nnue.as_secs_f64()
    } else {
        0.0
    };

    println!(
        "FH-NNUE | Best: {:?} | Score: {:?} | Nodes: {:>7} | Time: {:>5.1} ms | NPS: {:>9.0}",
        res_nnue.best_move,
        res_nnue.score,
        nodes_nnue,
        elapsed_nnue.as_secs_f64() * 1000.0,
        nps_nnue
    );

    println!("\n============================================================");
    println!("                 Benchmark Complete                         ");
    println!("============================================================");
}
