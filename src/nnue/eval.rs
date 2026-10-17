//! NNUE evaluation interface and score calculation.

use super::accumulator::PerspectiveAccumulators;
use super::model::NnueModel;
use crate::types::{Board, Color, Score};
use std::sync::{Arc, RwLock};

/// Global NNUE model storage for search threads.
static GLOBAL_MODEL: RwLock<Option<Arc<NnueModel>>> = RwLock::new(None);
static USE_NNUE: RwLock<bool> = RwLock::new(true);

/// Load a global NNUE model from a binary `.nnue` file.
pub fn load_global_model<P: AsRef<std::path::Path>>(path: P) -> Result<(), super::model::ModelError> {
    let model = NnueModel::load_from_file(path)?;
    let mut lock = GLOBAL_MODEL.write().unwrap();
    *lock = Some(Arc::new(model));
    Ok(())
}

/// Set an in-memory NNUE model as global.
pub fn set_global_model(model: NnueModel) {
    let mut lock = GLOBAL_MODEL.write().unwrap();
    *lock = Some(Arc::new(model));
}

/// Get a clone of the global NNUE model Arc, if available.
pub fn get_global_model() -> Option<Arc<NnueModel>> {
    GLOBAL_MODEL.read().unwrap().clone()
}

/// Enable or disable NNUE globally (switching to/from HCE).
pub fn set_use_nnue(enabled: bool) {
    let mut lock = USE_NNUE.write().unwrap();
    *lock = enabled;
}

/// Check if NNUE is enabled and a model is loaded.
pub fn is_nnue_enabled() -> bool {
    let enabled = *USE_NNUE.read().unwrap();
    enabled && GLOBAL_MODEL.read().unwrap().is_some()
}

/// Evaluate a position using NNUE from perspective accumulators.
/// Returns score from the side-to-move perspective.
#[inline]
pub fn evaluate_from_accumulators(
    acc: &PerspectiveAccumulators,
    turn: Color,
    model: &NnueModel,
) -> Score {
    let (stm_acc, nstm_acc) = match turn {
        Color::White => (&acc.white, &acc.black),
        Color::Black => (&acc.black, &acc.white),
    };

    model.head.evaluate(stm_acc, nstm_acc)
}

/// Fast PSQT evaluation of a position from perspective accumulators (~2.5 ns).
#[inline]
pub fn evaluate_psqt_from_accumulators(
    acc: &PerspectiveAccumulators,
    turn: Color,
    model: &NnueModel,
) -> Option<Score> {
    let (stm_acc, nstm_acc) = match turn {
        Color::White => (&acc.white, &acc.black),
        Color::Black => (&acc.black, &acc.white),
    };

    model.head.evaluate_psqt(stm_acc, nstm_acc)
}

/// Full fresh evaluation of a board using the supplied model.
#[inline]
pub fn evaluate_board(board: &Board, model: &NnueModel) -> Score {
    let psqt_weights = model.psqt_weights.as_deref();
    let acc = PerspectiveAccumulators::refresh(
        board,
        &model.feature_biases,
        &model.feature_weights,
        &model.psqt_biases,
        psqt_weights,
    );
    evaluate_from_accumulators(&acc, board.turn(), model)
}
