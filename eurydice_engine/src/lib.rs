pub mod diagnostic;
pub mod engine;
pub mod output;
pub mod primitives;

mod ast;
mod dice;
mod error;
mod eval;
mod operators;
mod value;

pub use ast::Range;
pub use diagnostic::{
    DiagnosticCode, DiagnosticLabel, DiagnosticSeverity, DiagnosticSource, EngineDiagnostic,
    EvaluationFrame, FixApplicability, LabelStyle, SourceId, SourceRange, SuggestedFix, TextEdit,
    TraceBinding,
};
pub use engine::{Engine, EngineOutput, Program, RunReport};
pub use output::{Distribution, FieldSchema};
pub use primitives::{PrimitiveMetadata, primitive_metadata};

use lalrpop_util::lalrpop_mod;
lalrpop_mod!(
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss
    )]
    grammar
);
