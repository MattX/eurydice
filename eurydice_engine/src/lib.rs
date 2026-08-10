mod diagnostic;
mod engine;
mod output;
mod primitives;

mod ast;
mod dice;
mod error;
mod eval;
mod operators;
mod value;

pub use ast::ByteRange;
pub use diagnostic::{
    Diagnostic, DiagnosticCode, DiagnosticLabel, DiagnosticSeverity, DiagnosticSource, Diagnostics,
    EvaluationFrame, SourceId, SourceRange, SuggestedFix, TraceBinding,
};
pub use engine::{Engine, EngineOutput, PrintEvent, Program, RunReport};
use lalrpop_util::lalrpop_mod;
pub use output::{Distribution, Field, FieldSchema};
pub use primitives::{PrimitiveMetadata, primitive_metadata};
lalrpop_mod!(
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss
    )]
    grammar
);
