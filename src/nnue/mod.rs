//! FH-NNUE: Factorized, Horizontally Canonicalized NNUE for Porcupine.
//!
//! Provides feature mapping, accumulators, quantized inference, model loading,
//! and evaluation integration.

pub mod accumulator;
pub mod eval;
pub mod features;
pub mod model;
pub mod network;

#[cfg(test)]
mod tests;

pub use accumulator::{Accumulator, PerspectiveAccumulators, TRANSFORMER_WIDTH};
pub use eval::{
    evaluate_board, evaluate_from_accumulators, evaluate_psqt_from_accumulators, get_global_model, is_nnue_enabled, load_global_model,
    set_global_model, set_use_nnue,
};
pub use features::{
    enumerate_features, FeatureIndex, PerspectiveOrientation, RelativePiece, NUM_FEATURES,
    NUM_KING_BUCKETS, NUM_PIECE_CATEGORIES, NUM_SQUARES,
};
pub use model::{ModelError, NnueModel};
pub use network::NetworkHead;
