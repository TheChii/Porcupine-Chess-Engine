use super::accumulator::PerspectiveAccumulators;
use super::features::*;
use super::model::NnueModel;
use crate::types::{Board, Color, Square};

#[test]
fn test_all_64_squares_canonical_mapping() {
    for sq_idx in 0..64 {
        let sq = unsafe { Square::from_index_unchecked(sq_idx) };
        // Test White perspective
        let w_orient = PerspectiveOrientation::from_king_square(Square::E1, Color::White);
        let w_mapped = w_orient.map_square(sq);
        assert!(w_mapped.index() < 64);

        // Test Black perspective
        let b_orient = PerspectiveOrientation::from_king_square(Square::E8, Color::Black);
        let b_mapped = b_orient.map_square(sq);
        assert!(b_mapped.index() < 64);
    }
}

#[test]
fn test_incremental_equals_refresh_startpos_moves() {
    let model = NnueModel::create_default();
    let board = Board::startpos();
    let psqt = model.psqt_weights.as_deref();

    let root_acc = PerspectiveAccumulators::refresh(
        &board,
        &model.feature_biases,
        &model.feature_weights,
        &model.psqt_biases,
        psqt,
    );

    let moves = board.generate_moves();
    for &m in moves.as_slice() {
        let new_board = board.make_move_new(m);

        // Compute incrementally
        let inc_acc = root_acc.update_move(
            &board,
            &new_board,
            m,
            &model.feature_biases,
            &model.feature_weights,
            &model.psqt_biases,
            psqt,
        );

        // Compute freshly from scratch
        let fresh_acc = PerspectiveAccumulators::refresh(
            &new_board,
            &model.feature_biases,
            &model.feature_weights,
            &model.psqt_biases,
            psqt,
        );

        assert_eq!(
            inc_acc.white.vals, fresh_acc.white.vals,
            "White accumulator mismatch after move {}",
            m
        );
        assert_eq!(
            inc_acc.black.vals, fresh_acc.black.vals,
            "Black accumulator mismatch after move {}",
            m
        );
    }
}

#[test]
fn test_incremental_equals_refresh_deep_game() {
    // Generate mock model with deterministic pseudo-random weights
    let mut model = NnueModel::create_default();
    for i in 0..NUM_FEATURES {
        for j in 0..16 {
            // Fill a few weights with non-zero values
            model.feature_weights[i][j] = ((i * 31 + j * 17) % 50) as i16;
        }
    }
    let psqt = model.psqt_weights.as_deref();

    let mut board = Board::startpos();
    let mut acc = PerspectiveAccumulators::refresh(
        &board,
        &model.feature_biases,
        &model.feature_weights,
        &model.psqt_biases,
        psqt,
    );

    // Play 30 deterministic pseudo-random legal moves
    for step in 0..30 {
        let moves = board.generate_moves();
        if moves.is_empty() {
            break;
        }
        // Pick move deterministically
        let m = moves.as_slice()[step % moves.len()];
        let new_board = board.make_move_new(m);

        let inc_acc = acc.update_move(
            &board,
            &new_board,
            m,
            &model.feature_biases,
            &model.feature_weights,
            &model.psqt_biases,
            psqt,
        );

        let fresh_acc = PerspectiveAccumulators::refresh(
            &new_board,
            &model.feature_biases,
            &model.feature_weights,
            &model.psqt_biases,
            psqt,
        );

        assert_eq!(
            inc_acc.white.vals, fresh_acc.white.vals,
            "White accumulator mismatch at ply {} after move {}",
            step, m
        );
        assert_eq!(
            inc_acc.black.vals, fresh_acc.black.vals,
            "Black accumulator mismatch at ply {} after move {}",
            step, m
        );

        board = new_board;
        acc = inc_acc;
    }
}

#[test]
fn test_special_moves_incremental() {
    let mut model = NnueModel::create_default();
    for i in 0..NUM_FEATURES {
        for j in 0..8 {
            model.feature_weights[i][j] = ((i + j) % 20) as i16;
        }
    }
    let psqt = model.psqt_weights.as_deref();

    // 1. Castling position (White O-O and Black O-O-O)
    let fen_castling = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
    let board = Board::from_fen(fen_castling).unwrap();
    let acc = PerspectiveAccumulators::refresh(
        &board,
        &model.feature_biases,
        &model.feature_weights,
        &model.psqt_biases,
        psqt,
    );

    let moves = board.generate_moves();
    for &m in moves.as_slice() {
        let new_board = board.make_move_new(m);
        let inc_acc = acc.update_move(
            &board,
            &new_board,
            m,
            &model.feature_biases,
            &model.feature_weights,
            &model.psqt_biases,
            psqt,
        );
        let fresh_acc = PerspectiveAccumulators::refresh(
            &new_board,
            &model.feature_biases,
            &model.feature_weights,
            &model.psqt_biases,
            psqt,
        );
        assert_eq!(inc_acc.white.vals, fresh_acc.white.vals);
        assert_eq!(inc_acc.black.vals, fresh_acc.black.vals);
    }

    // 2. En Passant position
    let fen_ep = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2";
    let board_ep = Board::from_fen(fen_ep).unwrap();
    let acc_ep = PerspectiveAccumulators::refresh(
        &board_ep,
        &model.feature_biases,
        &model.feature_weights,
        &model.psqt_biases,
        psqt,
    );
    let moves_ep = board_ep.generate_moves();
    for &m in moves_ep.as_slice() {
        let new_board = board_ep.make_move_new(m);
        let inc_acc = acc_ep.update_move(
            &board_ep,
            &new_board,
            m,
            &model.feature_biases,
            &model.feature_weights,
            &model.psqt_biases,
            psqt,
        );
        let fresh_acc = PerspectiveAccumulators::refresh(
            &new_board,
            &model.feature_biases,
            &model.feature_weights,
            &model.psqt_biases,
            psqt,
        );
        assert_eq!(inc_acc.white.vals, fresh_acc.white.vals);
        assert_eq!(inc_acc.black.vals, fresh_acc.black.vals);
    }

    // 3. Promotion position
    let fen_promo = "8/4P3/8/8/8/8/8/4K2k w - - 0 1";
    let board_promo = Board::from_fen(fen_promo).unwrap();
    let acc_promo = PerspectiveAccumulators::refresh(
        &board_promo,
        &model.feature_biases,
        &model.feature_weights,
        &model.psqt_biases,
        psqt,
    );
    let moves_promo = board_promo.generate_moves();
    for &m in moves_promo.as_slice() {
        let new_board = board_promo.make_move_new(m);
        let inc_acc = acc_promo.update_move(
            &board_promo,
            &new_board,
            m,
            &model.feature_biases,
            &model.feature_weights,
            &model.psqt_biases,
            psqt,
        );
        let fresh_acc = PerspectiveAccumulators::refresh(
            &new_board,
            &model.feature_biases,
            &model.feature_weights,
            &model.psqt_biases,
            psqt,
        );
        assert_eq!(inc_acc.white.vals, fresh_acc.white.vals);
        assert_eq!(inc_acc.black.vals, fresh_acc.black.vals);
    }
}

#[test]
fn test_load_and_evaluate_exported_model() {
    if std::path::Path::new("default.nnue").exists() {
        let model = NnueModel::load_from_file("default.nnue").expect("Failed to load exported default.nnue");
        let board = Board::startpos();
        let score = super::eval::evaluate_board(&board, &model);
        println!("Startpos evaluated score with loaded model: {:?}", score);
        assert!(score.raw().abs() < 10000);
    }
}

#[test]
fn test_model_rejection_malformed() {
    use super::model::ModelError;

    // 1. Truncated header
    let res = NnueModel::from_bytes(&[0u8; 10]);
    assert!(matches!(res, Err(ModelError::TruncatedFile { .. })));

    // 2. Bad magic
    let mut header = [0u8; 44];
    header[0..4].copy_from_slice(b"BADM");
    let res = NnueModel::from_bytes(&header);
    assert!(matches!(res, Err(ModelError::InvalidMagic(_))));

    // 3. Valid header but truncated payload
    let mut valid_header = [0u8; 44];
    valid_header[0..4].copy_from_slice(b"FNUE");
    valid_header[4..8].copy_from_slice(&1u32.to_le_bytes()); // format version
    valid_header[8..12].copy_from_slice(&1u32.to_le_bytes()); // feature version
    valid_header[12..16].copy_from_slice(&1u32.to_le_bytes()); // arch id
    valid_header[16..20].copy_from_slice(&0u32.to_le_bytes()); // no psqt
    valid_header[20..24].copy_from_slice(&(super::features::NUM_FEATURES as u32).to_le_bytes());
    valid_header[24..28].copy_from_slice(&(super::accumulator::TRANSFORMER_WIDTH as u32).to_le_bytes());
    valid_header[28..32].copy_from_slice(&(super::network::DENSE1_OUT as u32).to_le_bytes());
    valid_header[32..36].copy_from_slice(&(super::network::DENSE2_OUT as u32).to_le_bytes());
    valid_header[36..40].copy_from_slice(&(super::accumulator::PSQT_CHANNELS as u32).to_le_bytes());
    valid_header[40..44].copy_from_slice(&0x12345678u32.to_le_bytes()); // checksum

    let res = NnueModel::from_bytes(&valid_header);
    assert!(matches!(res, Err(ModelError::TruncatedFile { .. })));
}

#[test]
fn test_relative_displacement_bounds() {
    for k_sq_idx in 0..64 {
        let k_sq = unsafe { Square::from_index_unchecked(k_sq_idx) };
        let orient = PerspectiveOrientation::from_king_square(k_sq, Color::White);
        for p_sq_idx in 0..64 {
            let p_sq = unsafe { Square::from_index_unchecked(p_sq_idx) };
            let disp = orient.relative_displacement_index(p_sq);
            assert!(disp < NUM_REL_DISPLACEMENTS, "Displacement {} >= {}", disp, NUM_REL_DISPLACEMENTS);
        }
    }
}

#[test]
fn test_accumulator_saturation_overflow_safety() {
    let mut acc = super::accumulator::Accumulator::default();
    let extreme_weights = vec![[i16::MAX; super::accumulator::TRANSFORMER_WIDTH]; 1];

    // Adding multiple times saturates rather than overflowing/wrapping
    acc.add_feature(0, &extreme_weights, None);
    acc.add_feature(0, &extreme_weights, None);
    for i in 0..super::accumulator::TRANSFORMER_WIDTH {
        assert_eq!(acc.vals[i], i16::MAX);
    }

    // Subtracting multiple times saturates at i16::MIN
    acc.sub_feature(0, &extreme_weights, None);
    acc.sub_feature(0, &extreme_weights, None);
    acc.sub_feature(0, &extreme_weights, None);
    for i in 0..super::accumulator::TRANSFORMER_WIDTH {
        assert_eq!(acc.vals[i], i16::MIN);
    }
}

#[test]
fn test_search_accumulator_unmake_restoration() {
    let model = NnueModel::create_default();
    let board = Board::startpos();
    let psqt = model.psqt_weights.as_deref();

    let root_acc = PerspectiveAccumulators::refresh(
        &board,
        &model.feature_biases,
        &model.feature_weights,
        &model.psqt_biases,
        psqt,
    );

    let mut stack = [PerspectiveAccumulators::default(); 4];
    stack[0] = root_acc;

    let moves = board.generate_moves();
    let m1 = moves.as_slice()[0];
    let b1 = board.make_move_new(m1);
    stack[1] = stack[0].update_move(&board, &b1, m1, &model.feature_biases, &model.feature_weights, &model.psqt_biases, psqt);

    let moves1 = b1.generate_moves();
    let m2 = moves1.as_slice()[0];
    let b2 = b1.make_move_new(m2);
    stack[2] = stack[1].update_move(&b1, &b2, m2, &model.feature_biases, &model.feature_weights, &model.psqt_biases, psqt);

    // Verify stack[0] is still identical to root_acc (unmake preservation)
    assert_eq!(stack[0].white.vals, root_acc.white.vals);
    assert_eq!(stack[0].black.vals, root_acc.black.vals);
}

#[test]
fn test_search_nnue_exclusive_dispatch() {
    use std::sync::atomic::Ordering;
    use crate::eval::{reset_eval_counters, NNUE_EVAL_COUNT, HCE_EVAL_COUNT};
    use crate::search::{Searcher, SearchLimits};
    use crate::types::Depth;

    if std::path::Path::new("default.nnue").exists() {
        super::eval::load_global_model("default.nnue").expect("Failed to load default.nnue");

        // 1. Test NNUE Enabled
        super::eval::set_use_nnue(true);
        reset_eval_counters();

        let mut searcher = Searcher::new();
        searcher.set_position(Board::startpos());
        let mut limits = SearchLimits::default();
        limits.depth = Some(Depth::new(4));
        let _res = searcher.search(limits);

        let nnue_calls = NNUE_EVAL_COUNT.load(Ordering::Relaxed);
        let hce_calls = HCE_EVAL_COUNT.load(Ordering::Relaxed);

        println!("NNUE Enabled Search: NNUE calls = {}, HCE calls = {}", nnue_calls, hce_calls);
        assert!(nnue_calls > 0, "NNUE must be called during search when UseNNUE=true");
        assert_eq!(hce_calls, 0, "HCE must NOT be called when UseNNUE=true");

        // 2. Test HCE Fallback when NNUE Disabled
        super::eval::set_use_nnue(false);
        reset_eval_counters();

        let mut searcher_hce = Searcher::new();
        searcher_hce.set_position(Board::startpos());
        let mut limits_hce = SearchLimits::default();
        limits_hce.depth = Some(Depth::new(4));
        let _res_hce = searcher_hce.search(limits_hce);

        let nnue_calls_hce = NNUE_EVAL_COUNT.load(Ordering::Relaxed);
        let hce_calls_hce = HCE_EVAL_COUNT.load(Ordering::Relaxed);

        println!("HCE Search: NNUE calls = {}, HCE calls = {}", nnue_calls_hce, hce_calls_hce);
        assert_eq!(nnue_calls_hce, 0, "NNUE must NOT be called when UseNNUE=false");
        assert!(hce_calls_hce > 0, "HCE must be called when UseNNUE=false");

        // Restore default
        super::eval::set_use_nnue(true);
    }
}


