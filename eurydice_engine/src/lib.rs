pub mod ast;
pub mod diagnostic;
pub mod dice;
pub mod engine;
pub mod error;
pub mod eval;
pub mod output;
pub mod primitives;
pub mod value;

mod operators;

pub use diagnostic::{
    DiagnosticDetails, DiagnosticLabel, DiagnosticSeverity, DiagnosticSource, EngineDiagnostic,
    EvaluationFrame, FixApplicability, LabelStyle, SourceId, SourceRange, SuggestedFix, TextEdit,
    TraceBinding, ValueSummary,
};
pub use engine::{Engine, EngineError, EngineOutput, RunReport};
pub use primitives::{PrimitiveMetadata, primitive_metadata};

use lalrpop_util::lalrpop_mod;
lalrpop_mod!(
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_precision_loss)]
    pub grammar
);
