//! Testing utilities (only available with std feature).

mod perft;
mod positions;

pub use perft::{perft, perft_divide};
pub use positions::*;
