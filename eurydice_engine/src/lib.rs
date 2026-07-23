pub mod ast;
pub mod dice;
pub mod engine;
pub mod error;
pub mod eval;
pub mod output;
pub mod primitives;
pub mod value;

mod operators;

pub use engine::{Engine, EngineError, EngineOutput};

use lalrpop_util::lalrpop_mod;
lalrpop_mod!(
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_precision_loss)]
    pub grammar
);
