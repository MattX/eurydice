//! Serializable, frontend-neutral diagnostics produced by the engine.

use lalrpop_util::ParseError;
use miette::SourceSpan;
use serde::Serialize;

use crate::{
    ast::{FunctionDefinition, ParseActionError, Range, Statement, WithRange},
    error::{
        ArityMismatch, NonAdditiveSubject, OutcomeConflict, OutcomeMismatchContext,
        PrimitiveArgumentErrorKind, RuntimeError, SemanticErrorKind,
    },
    primitives::primitive_signature,
    value::{ElementType, ElementValue, RuntimeValue, SymbolTable},
};

/// Identifies one source submission within a stateful [`crate::Engine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct SourceId(pub u64);

/// Source text referenced by a diagnostic.
#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticSource {
    pub id: SourceId,
    pub name: String,
    pub text: String,
}

/// A range whose source submission is explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct SourceRange {
    pub source: SourceId,
    pub range: Range,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    Error,
    Warning,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelStyle {
    Primary,
    Secondary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiagnosticLabel {
    pub range: SourceRange,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    pub style: LabelStyle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FixApplicability {
    MachineApplicable,
    Suggested,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TextEdit {
    pub range: SourceRange,
    pub replacement: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SuggestedFix {
    pub message: String,
    pub applicability: FixApplicability,
    pub edits: Vec<TextEdit>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TraceBinding {
    pub name: String,
    /// The bound value, rendered for display.
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvaluationFrame {
    pub function: String,
    pub call: SourceRange,
    pub definition: Option<SourceRange>,
    pub bindings: Vec<TraceBinding>,
}

/// A complete engine diagnostic.
///
/// The `code` classifies the diagnostic for consumers that want to branch on
/// it; everything else is prose and spans for them to present.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EngineDiagnostic {
    pub code: String,
    pub severity: DiagnosticSeverity,
    pub summary: String,
    pub labels: Vec<DiagnosticLabel>,
    pub notes: Vec<String>,
    pub help: Option<String>,
    pub fixes: Vec<SuggestedFix>,
    pub trace: Vec<EvaluationFrame>,
    pub incomplete: bool,
}

impl EngineDiagnostic {
    pub fn primary_range(&self) -> Option<SourceRange> {
        self.labels
            .iter()
            .find(|label| label.style == LabelStyle::Primary)
            .or_else(|| self.labels.first())
            .map(|label| label.range)
    }
}

/// A value rendered for display, truncated so one oversized pool cannot push
/// the rest of a diagnostic off the screen.
pub(crate) fn preview_value(value: &RuntimeValue, symbols: &SymbolTable) -> String {
    let preview = value.display(symbols).to_string();
    if preview.chars().count() > 80 {
        format!("{}…", preview.chars().take(79).collect::<String>())
    } else {
        preview
    }
}

fn describe_value(value: &RuntimeValue, symbols: &SymbolTable) -> String {
    match value {
        RuntimeValue::Element(ElementValue::AdditiveIdentity | ElementValue::Int(_)) => {
            "an integer".to_string()
        }
        RuntimeValue::Element(ElementValue::Symbol(symbol)) => {
            format!("a `{}` value", symbols.set_of(*symbol).name)
        }
        RuntimeValue::Element(ElementValue::Tuple(_)) => "a tuple".to_string(),
        RuntimeValue::List(values, outcome_type) => format!(
            "a sequence of `{}` values",
            describe_outcomes(values.iter(), outcome_type, symbols)
        ),
        RuntimeValue::Pool(pool, outcome_type) => format!(
            "a dice pool with `{}` outcomes",
            describe_outcomes(
                pool.ordered_outcomes().iter().map(|(outcome, _)| outcome),
                outcome_type,
                symbols
            )
        ),
    }
}

/// Names a collection's outcome kinds for a diagnostic.
///
/// The outcome type only records whether every value is an `int`, so anything
/// more specific — which declared sets a collection's symbols come from —
/// is read back off the values themselves.
fn describe_outcomes<'a>(
    values: impl IntoIterator<Item = &'a ElementValue>,
    outcome_type: &ElementType,
    symbols: &SymbolTable,
) -> String {
    if !matches!(outcome_type, ElementType::Mixed) {
        return outcome_type.display_name();
    }
    let mut sets: Vec<&str> = Vec::new();
    let mut has_int = false;
    for value in values {
        match value {
            ElementValue::Symbol(symbol) => {
                let name = symbols.set_of(*symbol).name.as_str();
                if !sets.contains(&name) {
                    sets.push(name);
                }
            }
            _ => has_int = true,
        }
    }
    if has_int {
        sets.insert(0, "int");
    }
    match sets.len() {
        0 => outcome_type.display_name(),
        _ => sets.join(" or "),
    }
}

/// How a value is shaped, which is the only thing two outcomes of one
/// collection can irreconcilably disagree about.
fn shape_name(outcome_type: &ElementType) -> String {
    match outcome_type {
        ElementType::Tuple(fields) => format!("a tuple of {} fields", fields.len()),
        ElementType::AdditiveIdentity | ElementType::Uninhabited => "the empty sum".to_string(),
        ElementType::Int | ElementType::Mixed => "a single value".to_string(),
    }
}

fn primitive_call(identifier: &str) -> String {
    primitive_signature(identifier)
        .map(str::to_string)
        .unwrap_or_else(|| format!("[{}]", identifier.replace("{}", "…")))
}

/// A primary label reporting the value found where another type was required.
fn type_mismatch_label(
    source: SourceId,
    range: &SourceSpan,
    expected: &str,
    value: &RuntimeValue,
    symbols: &SymbolTable,
) -> DiagnosticLabel {
    DiagnosticLabel {
        range: SourceRange {
            source,
            range: range.into(),
        },
        message: Some(format!(
            "`{}` is {}; expected {expected}",
            preview_value(value, symbols),
            describe_value(value, symbols)
        )),
        style: LabelStyle::Primary,
    }
}

/// The same report for a named argument, which leads with the name because a
/// call can have more than one argument at fault.
fn argument_mismatch_message(
    name: &str,
    value: &RuntimeValue,
    expected: &str,
    symbols: &SymbolTable,
) -> String {
    format!(
        "`{name}` is {}: `{}`; expected {expected}",
        describe_value(value, symbols),
        preview_value(value, symbols)
    )
}

struct DiagnosticParts {
    code: &'static str,
    summary: String,
    labels: Vec<DiagnosticLabel>,
    notes: Vec<String>,
    help: Option<String>,
    fixes: Vec<SuggestedFix>,
    incomplete: bool,
}

impl DiagnosticParts {
    fn new(code: &'static str, summary: impl Into<String>, labels: Vec<DiagnosticLabel>) -> Self {
        Self {
            code,
            summary: summary.into(),
            labels,
            notes: Vec::new(),
            help: None,
            fixes: Vec::new(),
            incomplete: false,
        }
    }

    fn finish(self, severity: DiagnosticSeverity, trace: Vec<EvaluationFrame>) -> EngineDiagnostic {
        EngineDiagnostic {
            code: self.code.to_string(),
            severity,
            summary: self.summary,
            labels: self.labels,
            notes: self.notes,
            help: self.help,
            fixes: self.fixes,
            trace,
            incomplete: self.incomplete,
        }
    }

    fn help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    fn maybe_help(mut self, help: Option<String>) -> Self {
        self.help = help;
        self
    }

    fn note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    fn incomplete(mut self) -> Self {
        self.incomplete = true;
        self
    }

    fn fix(mut self, fix: SuggestedFix) -> Self {
        self.fixes.push(fix);
        self
    }
}

/// A syntax diagnostic with a single unlabeled primary span.
fn syntax_parts(
    source: SourceId,
    range: Range,
    code: &'static str,
    summary: impl Into<String>,
) -> DiagnosticParts {
    DiagnosticParts::new(
        code,
        summary,
        vec![DiagnosticLabel {
            range: SourceRange { source, range },
            message: None,
            style: LabelStyle::Primary,
        }],
    )
}

pub(crate) fn parse_error<T: std::fmt::Display>(
    error: ParseError<usize, T, ParseActionError>,
    source_id: SourceId,
    source: &str,
) -> EngineDiagnostic {
    let parts = match error {
        ParseError::UnrecognizedToken {
            token, expected, ..
        } => {
            let range = Range::from((token.0, token.2));
            let found = source.get(token.0..token.2).unwrap_or_default();
            if found == "=" && expected.iter().any(|item| item.contains(':')) {
                syntax_parts(
                    source_id,
                    range,
                    "syntax.assignment_separator",
                    "Assignments use `:`, not `=`",
                )
                .help("Write `NAME: value` to bind a variable.")
                .fix(replacement_fix(
                    source_id,
                    range,
                    ":",
                    "Replace `=` with `:`",
                ))
            } else if found == "="
                && source
                    .get(..token.0)
                    .is_some_and(|prefix| prefix.ends_with('='))
            {
                syntax_parts(
                    source_id,
                    range,
                    "syntax.equality_operator",
                    "Equality uses one `=`",
                )
                .help("AnyDice writes equality as `A = B`.")
                .fix(replacement_fix(
                    source_id,
                    range,
                    "",
                    "Remove the second `=`",
                ))
            } else if is_plain_word(found) && expected.iter().any(|item| item.contains("[A-Z_]")) {
                // A lowercase word where a value was expected: lowercase words
                // only ever name functions, so this reads as a variable whose
                // case is wrong.
                let replacement = found.to_ascii_uppercase();
                syntax_parts(
                    source_id,
                    range,
                    "syntax.variable_case",
                    "Variable names use uppercase letters",
                )
                .help("Lowercase words are reserved for function names.")
                .fix(replacement_fix(
                    source_id,
                    range,
                    &replacement,
                    &format!("Change this variable to `{replacement}`"),
                ))
            } else {
                syntax_parts(
                    source_id,
                    range,
                    "syntax.unexpected_token",
                    format!("Unexpected token `{found}`"),
                )
                .note("At the top level, expressions you want to show must start with `output`.")
            }
        }
        ParseError::UnrecognizedEof { location, expected } => {
            let range = Range::from((location, location));
            if let Some(closer) = expected_closer(&expected) {
                syntax_parts(
                    source_id,
                    range,
                    "syntax.unclosed_delimiter",
                    format!("Missing closing `{closer}`"),
                )
                .incomplete()
                .help(format!(
                    "Add the matching `{closer}` to complete this expression."
                ))
                .fix(replacement_fix(
                    source_id,
                    range,
                    &closer.to_string(),
                    &format!("Insert `{closer}`"),
                ))
            } else {
                syntax_parts(
                    source_id,
                    range,
                    "syntax.unexpected_end",
                    "The program ends before this construct is complete",
                )
                .incomplete()
            }
        }
        ParseError::ExtraToken { token } => {
            let range = Range::from((token.0, token.2));
            syntax_parts(
                source_id,
                range,
                "syntax.extra_token",
                format!(
                    "Unexpected extra token `{}`",
                    source.get(token.0..token.2).unwrap_or_default()
                ),
            )
        }
        ParseError::InvalidToken { location } => {
            let (end, found) = next_char(source, location);
            let range = Range::from((location, end));
            let insertion = Range {
                start: source.len(),
                end: source.len(),
            };
            if found == Some('"') && !has_unescaped_quote(&source[end..]) {
                syntax_parts(
                    source_id,
                    range,
                    "syntax.unterminated_string",
                    "This string is missing its closing quote",
                )
                .incomplete()
                .help("String literals end with `\"`.")
                .fix(replacement_fix(
                    source_id,
                    insertion,
                    "\"",
                    "Insert the closing quote",
                ))
            } else if found == Some('\\') && !source[end..].contains('\\') {
                syntax_parts(
                    source_id,
                    range,
                    "syntax.unterminated_comment",
                    "This block comment is missing its closing backslash",
                )
                .incomplete()
                .help("AnyDice block comments are enclosed by backslashes.")
                .fix(replacement_fix(
                    source_id,
                    insertion,
                    "\\",
                    "Insert the closing backslash",
                ))
            } else if found == Some(';') {
                syntax_parts(
                    source_id,
                    range,
                    "syntax.semicolon",
                    "Statements are not separated with semicolons",
                )
                .help("Whitespace is enough to separate AnyDice statements.")
                .fix(replacement_fix(
                    source_id,
                    range,
                    "",
                    "Remove the semicolon",
                ))
            } else {
                syntax_parts(
                    source_id,
                    range,
                    "syntax.invalid_character",
                    found.map_or_else(
                        || "Invalid character".to_string(),
                        |character| format!("Invalid character `{character}`"),
                    ),
                )
            }
        }
        ParseError::User { error } => match error {
            ParseActionError::InvalidIntegerLiteral { range, .. } => syntax_parts(
                source_id,
                range,
                "syntax.integer_out_of_range",
                "This integer is outside the supported range",
            )
            .help("Integers must fit between -2147483648 and 2147483647."),
            ParseActionError::EmptyFunctionCall { range } => syntax_parts(
                source_id,
                range,
                "syntax.empty_function_call",
                "A function call cannot be empty",
            )
            .help("Put a lowercase function name between the brackets."),
        },
    };

    parts.finish(DiagnosticSeverity::Error, Vec::new())
}

/// Whether `word` is a lowercase word that carries no meaning of its own, and
/// so can only be a name. Keywords are excluded: misplacing one is its own
/// mistake, and uppercasing it would not be the fix.
fn is_plain_word(word: &str) -> bool {
    const KEYWORDS: [&str; 17] = [
        "d", "if", "set", "else", "to", "function", "result", "over", "loop", "output", "print",
        "named", "labeled", "n", "s", "int", "enum",
    ];

    !word.is_empty()
        && word.starts_with(|c: char| c.is_ascii_lowercase())
        && word.chars().all(|c| c.is_ascii_lowercase() || c == '_')
        && !KEYWORDS.contains(&word)
}

/// A warning the evaluator can raise.
///
/// Each variant owns its diagnostic code and doubles as the identity used to
/// report a warning only once, so the two can never drift apart. Codes that
/// differ need separate variants, or one would suppress the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum WarningKind {
    IndependentPoolReuse,
    ExplodeDepth,
    MaximumFunctionDepth,
    MissingResult,
}

impl WarningKind {
    pub(crate) fn code(self) -> &'static str {
        match self {
            Self::IndependentPoolReuse => "evaluation.independent_pool_reuse",
            Self::ExplodeDepth => "evaluation.explode_depth",
            Self::MaximumFunctionDepth => "evaluation.maximum_function_depth",
            Self::MissingResult => "control_flow.missing_result",
        }
    }
}

pub(crate) fn missing_return_warning(
    source: SourceId,
    definition: &FunctionDefinition,
) -> Option<EngineDiagnostic> {
    if block_always_returns(&definition.body) {
        return None;
    }

    let function = definition.name.value.replace("{}", "…");
    Some(EngineDiagnostic {
        code: WarningKind::MissingResult.code().to_string(),
        severity: DiagnosticSeverity::Warning,
        summary: format!("Function `[{function}]` may finish without a result"),
        labels: vec![DiagnosticLabel {
            range: SourceRange {
                source,
                range: definition.name.range,
            },
            message: Some("not every path reaches `result:`".to_string()),
            style: LabelStyle::Primary,
        }],
        notes: vec![
            "When execution reaches the end of a function, the function produces an empty die."
                .to_string(),
        ],
        help: Some("Add `result:` on every path through this function.".to_string()),
        fixes: Vec::new(),
        trace: Vec::new(),
        incomplete: false,
    })
}

fn block_always_returns(block: &[WithRange<Statement>]) -> bool {
    block.iter().any(|statement| match &statement.value {
        Statement::Return { .. } => true,
        Statement::If {
            then_block,
            else_block: Some(else_block),
            ..
        } => block_always_returns(then_block) && block_always_returns(else_block),
        // A loop body that always returns only returns when the sequence is
        // non-empty, which cannot be known here. Counting it keeps "return the
        // first element of S" — an ordinary idiom — from being warned about;
        // missing the empty-sequence case costs less than crying wolf.
        Statement::Loop { body, .. } => block_always_returns(body),
        _ => false,
    })
}

fn replacement_fix(
    source: SourceId,
    range: Range,
    replacement: &str,
    message: &str,
) -> SuggestedFix {
    SuggestedFix {
        message: message.to_string(),
        applicability: FixApplicability::MachineApplicable,
        edits: vec![TextEdit {
            range: SourceRange { source, range },
            replacement: replacement.to_string(),
        }],
    }
}

fn next_char(source: &str, location: usize) -> (usize, Option<char>) {
    let character = source.get(location..).and_then(|rest| rest.chars().next());
    (
        character.map_or(location, |character| location + character.len_utf8()),
        character,
    )
}

/// The single delimiter that would let the program continue at end of input.
///
/// The parse tables already track open constructs, so the closer is read from
/// what the parser expected rather than by rescanning the source — that keeps
/// the grammar the only thing that tokenizes Eurydice, and it stays right about
/// delimiters inside strings and comments for free. `None` when the input can
/// continue in more ways than one closing delimiter, which is the ordinary
/// "ran out of input" case rather than an unclosed bracket.
fn expected_closer(expected: &[String]) -> Option<char> {
    let mut closers = expected.iter().filter_map(|token| match token.as_str() {
        r#"")""# => Some(')'),
        r#""]""# => Some(']'),
        r#""}""# => Some('}'),
        _ => None,
    });
    let closer = closers.next()?;
    closers.next().is_none().then_some(closer)
}

fn has_unescaped_quote(source: &str) -> bool {
    let mut escaped = false;
    for character in source.chars() {
        if escaped {
            escaped = false;
        } else if character == '\\' {
            escaped = true;
        } else if character == '"' {
            return true;
        }
    }
    false
}

/// Renders a runtime error. `suggestions` holds near-miss names from the
/// namespace the error looked in, and is empty for errors that name nothing.
pub(crate) fn runtime_diagnostic(
    error: &RuntimeError,
    source_id: SourceId,
    suggestions: &[String],
    symbols: &SymbolTable,
) -> EngineDiagnostic {
    let mut trace = Vec::new();
    let (error, source_id) = runtime_context(error, source_id, &mut trace);
    let primary = |range: &SourceSpan, message: String| DiagnosticLabel {
        range: SourceRange {
            source: source_id,
            range: range.into(),
        },
        message: Some(message),
        style: LabelStyle::Primary,
    };
    let unlabeled = |range: &SourceSpan, style: LabelStyle| DiagnosticLabel {
        range: SourceRange {
            source: source_id,
            range: range.into(),
        },
        message: None,
        style,
    };

    let parts = match error {
        RuntimeError::UndefinedReference { range, name } => DiagnosticParts::new(
            "name.undefined_variable",
            format!("Variable `{name}` is not defined"),
            vec![unlabeled(range, LabelStyle::Primary)],
        )
        .help(suggestion_help(
            "Variable names are uppercase and must be assigned before use.",
            suggestions,
        )),
        RuntimeError::UndefinedFunction {
            range,
            name,
            arity_mismatch,
        } => {
            let parts = DiagnosticParts::new(
                "name.undefined_function",
                format!("Function `[{}]` is not defined", name.replace("{}", "…")),
                vec![unlabeled(range, LabelStyle::Primary)],
            )
            .help(arity_mismatch.as_ref().map_or_else(
                || {
                    suggestion_help(
                        "Check the function's words and argument positions.",
                        suggestions,
                    )
                },
                arity_mismatch_help,
            ));
            match arity_mismatch.as_ref().and_then(|m| m.comma_insertion) {
                Some(offset) => parts.fix(SuggestedFix {
                    message: "Separate these dice arguments with a comma".to_string(),
                    applicability: FixApplicability::Suggested,
                    edits: vec![TextEdit {
                        range: SourceRange {
                            source: source_id,
                            range: Range {
                                start: offset,
                                end: offset,
                            },
                        },
                        replacement: ",".to_string(),
                    }],
                }),
                None => parts,
            }
        }
        RuntimeError::LoopOverNonSequence { range, value } => {
            let parts = DiagnosticParts::new(
                "type.expected_sequence",
                "A loop can only iterate over a sequence",
                vec![type_mismatch_label(
                    source_id,
                    range,
                    "a sequence",
                    value,
                    symbols,
                )],
            );
            if matches!(value, RuntimeValue::Pool(_, _)) {
                parts.help(
                    "A dice pool represents a distribution. Pass it through an `s` parameter to \
                     evaluate a function once for each possible roll.",
                )
            } else {
                parts.help("Write a sequence between `{` and `}`, such as `{1, 2, 3}`.")
            }
        }
        RuntimeError::InvalidCondition { range, value } => {
            let parts = DiagnosticParts::new(
                "type.expected_number",
                "An `if` condition must be an integer",
                vec![type_mismatch_label(
                    source_id,
                    range,
                    "an integer",
                    value,
                    symbols,
                )],
            );
            if matches!(value, RuntimeValue::Pool(_, _)) {
                parts.help(
                    "To branch on each possible die result, put the condition in a function with \
                     an `n` parameter and pass the die to it.",
                )
            } else {
                parts.help("Use a comparison such as `A = B` to produce `0` (false) or `1` (true).")
            }
        }
        RuntimeError::InvalidArgumentToOperator {
            operator_range,
            op,
            expected,
            found_range,
            value,
        } => DiagnosticParts::new(
            "type.operator_argument",
            format!("Operator `{op}` expects {expected}"),
            vec![
                unlabeled(operator_range, LabelStyle::Secondary),
                primary(
                    found_range,
                    format!(
                        "`{}` is {}",
                        preview_value(value, symbols),
                        describe_value(value, symbols)
                    ),
                ),
            ],
        ),
        RuntimeError::NegativeArgumentToFunction {
            range,
            name,
            found_range,
            value,
        } => DiagnosticParts::new(
            "value.nonnegative_required",
            format!("`{}` needs a nonnegative count", primitive_call(name)),
            vec![
                unlabeled(range, LabelStyle::Secondary),
                primary(found_range, format!("this evaluates to {value}")),
            ],
        ),
        RuntimeError::InvalidPrimitiveArguments(error) => {
            let labels = error
                .arguments
                .iter()
                .enumerate()
                .map(|(index, argument)| DiagnosticLabel {
                    range: SourceRange {
                        source: source_id,
                        range: (&argument.range).into(),
                    },
                    message: Some(argument_mismatch_message(
                        argument.name,
                        &argument.value,
                        &argument.expected,
                        symbols,
                    )),
                    style: if index == 0 {
                        LabelStyle::Primary
                    } else {
                        LabelStyle::Secondary
                    },
                })
                .collect::<Vec<_>>();
            let code = match error.kind {
                PrimitiveArgumentErrorKind::Type => "type.function_argument",
            };
            DiagnosticParts::new(
                code,
                format!("`{}` {}", primitive_call(error.function), error.requirement),
                labels,
            )
            .maybe_help(error.help.clone())
        }
        RuntimeError::InvalidPrimitiveValue(error) => DiagnosticParts::new(
            "value.out_of_range",
            format!("`{}` {}", primitive_call(error.function), error.requirement),
            vec![primary(
                &error.found_range,
                argument_mismatch_message(error.argument, &error.value, &error.constraint, symbols),
            )],
        )
        .maybe_help(error.help.clone()),
        RuntimeError::OutcomeMismatch(error) => {
            let subject = match error.context {
                OutcomeMismatchContext::SequenceLiteral => "A sequence",
                OutcomeMismatchContext::FunctionResults => "A function evaluated over a pool",
            };
            let first = shape_name(&error.first.outcome_type);
            let second = shape_name(&error.second.outcome_type);
            let empty_sum = |conflict: &OutcomeConflict| {
                matches!(
                    conflict.outcome_type,
                    ElementType::AdditiveIdentity | ElementType::Uninhabited
                )
            };
            // Numbers and symbols mix freely, so the only irreconcilable
            // difference is one of shape — or the empty sum, which takes the
            // shape of whatever it is added to and so needs something addable.
            let involves_empty_sum = empty_sum(&error.first) || empty_sum(&error.second);
            let summary = if involves_empty_sum {
                format!("{subject} cannot mix the empty sum with values that cannot be added")
            } else {
                format!("{subject} cannot mix {first} and {second}")
            };
            // Against the empty sum the shape is not the point — what matters
            // is that the other value cannot be added at all.
            let describe = |conflict: &OutcomeConflict, name: &str, settled: bool| match (
                involves_empty_sum,
                empty_sum(conflict),
                settled,
            ) {
                (true, false, _) => "this cannot be added".to_string(),
                (true, true, _) => "this is the empty sum".to_string(),
                (false, _, true) => format!("this is {name}, which set the shape"),
                (false, _, false) => format!("this is {name}"),
            };
            let labels = match (error.first.range, error.second.range) {
                (Some(first_range), Some(second_range)) => vec![
                    primary(&second_range, describe(&error.second, &second, false)),
                    DiagnosticLabel {
                        range: SourceRange {
                            source: source_id,
                            range: (&first_range).into(),
                        },
                        message: Some(describe(&error.first, &first, true)),
                        style: LabelStyle::Secondary,
                    },
                ],
                _ => vec![primary(
                    &error.range,
                    format!("one result is {first}, another is {second}"),
                )],
            };
            DiagnosticParts::new("type.outcome_mismatch", summary, labels).help(
                if involves_empty_sum {
                    "The empty sum stands in for a value of whatever type it is added to, so it \
                     can only share a collection with values that can be added."
                } else {
                    "Values sharing a sequence or a pool may be numbers or symbols, but they must \
                     all be single values, or all tuples of the same size."
                },
            )
        }
        RuntimeError::NonAdditiveSum(error) => {
            let (subject, parts_of_it) = match error.subject {
                NonAdditiveSubject::Pool(dimension) => {
                    (format!("a pool of {dimension} dice"), "outcomes")
                }
                NonAdditiveSubject::Sequence(length) => {
                    (format!("a sequence of {length} values"), "values")
                }
            };
            let label = match (&error.witness, error.field) {
                (Some(witness), Some(field)) => format!(
                    "field {field} of {parts_of_it} like `{}` is not a number",
                    witness.display(symbols)
                ),
                (Some(witness), None) => format!(
                    "{parts_of_it} like `{}` are not numbers",
                    witness.display(symbols)
                ),
                (None, _) => format!("these {parts_of_it} cannot be added together"),
            };
            DiagnosticParts::new(
                "type.non_additive_value",
                format!(
                    "{} requires summing {subject}, but they cannot be added together",
                    error.action
                ),
                vec![primary(&error.range, label)],
            )
            .help(match error.field {
                Some(_) => {
                    "Tuples are added field by field, so every field must be a number. Project the \
                     numeric fields with `[field N of ...]` first."
                }
                None => {
                    "Only numbers can be added. `[sum integers in ...]` totals just the numeric \
                     outcomes, and `print` displays a pool without summing it."
                }
            })
        }
        RuntimeError::MathError { range, message } => DiagnosticParts::new(
            "value.arithmetic",
            message,
            vec![unlabeled(range, LabelStyle::Primary)],
        ),
        RuntimeError::OutputNotAtTopLevel { range } => placement_diagnostic(
            "output",
            range,
            source_id,
            "Use `print` for debugging inside a function, or return the value with `result:`.",
        ),
        RuntimeError::SetNotAtTopLevel { range } => placement_diagnostic(
            "set",
            range,
            source_id,
            "Move this setting before the function definition.",
        ),
        RuntimeError::ReturnOutsideFunction { range } => DiagnosticParts::new(
            "placement.result_outside_function",
            "`result:` can only appear inside a function",
            vec![unlabeled(range, LabelStyle::Primary)],
        ),
        RuntimeError::LabelsOnNonTupleOutput {
            range,
            value_range,
            value,
        } => DiagnosticParts::new(
            "type.labels_require_tuple",
            "Output labels require a tuple-valued output",
            vec![
                unlabeled(range, LabelStyle::Secondary),
                type_mismatch_label(source_id, value_range, "a tuple", value, symbols),
            ],
        ),
        RuntimeError::OutputLabelCountMismatch {
            range,
            expected,
            found,
        } => DiagnosticParts::new(
            "value.output_label_count",
            format!(
                "This tuple has {expected} fields, but {found} output {} provided",
                if *found == 1 {
                    "label was"
                } else {
                    "labels were"
                }
            ),
            vec![unlabeled(range, LabelStyle::Primary)],
        ),
        RuntimeError::InvalidRepeatExpression { range, value } => DiagnosticParts::new(
            "type.repeat_count",
            "A repetition count must be an integer",
            vec![type_mismatch_label(
                source_id,
                range,
                "an integer",
                value,
                symbols,
            )],
        ),
        RuntimeError::RangeHasNonSequenceEndpoints { range, value } => DiagnosticParts::new(
            "type.range_endpoint",
            "Both ends of a range must be integers",
            vec![type_mismatch_label(
                source_id,
                range,
                "an integer",
                value,
                symbols,
            )],
        ),
        RuntimeError::Semantic {
            kind,
            range,
            message,
        } => DiagnosticParts::new(
            semantic_code(*kind),
            message,
            vec![unlabeled(range, LabelStyle::Primary)],
        ),
        RuntimeError::InFunction { .. } => {
            unreachable!("function contexts were peeled before rendering")
        }
    };

    parts.finish(DiagnosticSeverity::Error, trace)
}

/// The stable diagnostic code for a semantic error.
fn semantic_code(kind: SemanticErrorKind) -> &'static str {
    match kind {
        SemanticErrorKind::OperatorOperands => "type.operator_operands",
        SemanticErrorKind::OutcomeMismatch => "type.outcome_mismatch",
        SemanticErrorKind::NonAdditiveValue => "type.non_additive_value",
        SemanticErrorKind::OutOfRange => "value.out_of_range",
        SemanticErrorKind::BindingConflict => "name.binding_conflict",
        SemanticErrorKind::TopLevelOnly => "placement.top_level_only",
    }
}

fn arity_mismatch_help(mismatch: &ArityMismatch) -> String {
    let counts = match mismatch.available.as_slice() {
        [only] => format!("{only} argument{}", if *only == 1 { "" } else { "s" }),
        [rest @ .., last] => format!(
            "{} or {last} arguments",
            rest.iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        [] => unreachable!("an arity mismatch names at least one other arity"),
    };
    let described = if mismatch.words.is_empty() {
        "a function with no words".to_string()
    } else {
        format!("[{}]", mismatch.words)
    };
    format!(
        "{described} takes {counts}, but this call passes {}. If two arguments were joined into \
         one expression, separate them with a comma: `d` binds tighter than argument separation, \
         so `[f d6 d8]` passes one argument and `[f d6, d8]` passes two.",
        mismatch.found
    )
}

fn suggestion_help(base: &str, suggestions: &[String]) -> String {
    match suggestions {
        [] => base.to_string(),
        [only] => format!("{base} Did you mean `{only}`?"),
        many => format!("{base} Similar names: {}.", many.join(", ")),
    }
}

/// A name an error reports as undefined, and the namespace it was looked up in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UndefinedName<'a> {
    Variable(&'a str),
    Function(&'a str),
}

impl<'a> UndefinedName<'a> {
    pub(crate) fn name(self) -> &'a str {
        match self {
            Self::Variable(name) | Self::Function(name) => name,
        }
    }
}

pub(crate) fn undefined_name(error: &RuntimeError) -> Option<UndefinedName<'_>> {
    match error {
        RuntimeError::UndefinedReference { name, .. } => Some(UndefinedName::Variable(name)),
        RuntimeError::UndefinedFunction { name, .. } => Some(UndefinedName::Function(name)),
        RuntimeError::InFunction { source, .. } => undefined_name(source),
        _ => None,
    }
}

pub(crate) fn add_later_definition(
    diagnostic: &mut EngineDiagnostic,
    error: &RuntimeError,
    later: &[crate::ast::WithRange<crate::ast::Statement>],
    source: SourceId,
) {
    let Some(undefined) = undefined_name(error) else {
        return;
    };
    let name = undefined.name();
    let is_function = matches!(undefined, UndefinedName::Function(_));
    let definition = later.iter().find_map(|statement| match &statement.value {
        crate::ast::Statement::Assignment {
            name: candidate, ..
        } if !is_function && candidate.value == name => Some(candidate.range),
        crate::ast::Statement::FunctionDefinition(definition)
            if is_function && definition.name.value == name =>
        {
            Some(definition.name.range)
        }
        _ => None,
    });
    if let Some(range) = definition {
        diagnostic.labels.push(DiagnosticLabel {
            range: SourceRange { source, range },
            message: Some("defined later".to_string()),
            style: LabelStyle::Secondary,
        });
        diagnostic.notes.push(
            "Statements execute in order, so this definition is not available yet.".to_string(),
        );
    }
}

fn runtime_context<'a>(
    error: &'a RuntimeError,
    default_source: SourceId,
    frames: &mut Vec<crate::diagnostic::EvaluationFrame>,
) -> (&'a RuntimeError, SourceId) {
    match error {
        RuntimeError::InFunction {
            source,
            body_source,
            frame,
            ..
        } => {
            frames.push((**frame).clone());
            runtime_context(source, *body_source, frames)
        }
        other => (other, default_source),
    }
}

fn placement_diagnostic(
    statement: &str,
    range: &SourceSpan,
    source_id: SourceId,
    help: &str,
) -> DiagnosticParts {
    DiagnosticParts::new(
        "placement.top_level_only",
        format!("`{statement}` can only appear at the top level"),
        vec![DiagnosticLabel {
            range: SourceRange {
                source: source_id,
                range: range.into(),
            },
            message: None,
            style: LabelStyle::Primary,
        }],
    )
    .help(help)
}

#[cfg(test)]
mod tests {
    use super::{SemanticErrorKind, expected_closer, semantic_code};

    /// The codes are part of the diagnostic contract, so they are pinned here
    /// rather than left to follow whatever a raise site's wording happens to be.
    #[test]
    fn every_semantic_kind_has_a_stable_code() {
        let codes = [
            (
                SemanticErrorKind::OperatorOperands,
                "type.operator_operands",
            ),
            (SemanticErrorKind::OutcomeMismatch, "type.outcome_mismatch"),
            (
                SemanticErrorKind::NonAdditiveValue,
                "type.non_additive_value",
            ),
            (SemanticErrorKind::OutOfRange, "value.out_of_range"),
            (SemanticErrorKind::BindingConflict, "name.binding_conflict"),
            (SemanticErrorKind::TopLevelOnly, "placement.top_level_only"),
        ];

        for (kind, code) in codes {
            assert_eq!(semantic_code(kind), code, "{kind:?}");
        }
    }

    #[test]
    fn expected_closer_names_a_delimiter_only_when_it_is_the_only_way_on() {
        let closer = |tokens: &[&str]| {
            expected_closer(
                &tokens
                    .iter()
                    .map(|token| token.to_string())
                    .collect::<Vec<_>>(),
            )
        };

        assert_eq!(closer(&[r#"")""#]), Some(')'));
        // A closer among other continuations is still unambiguous.
        assert_eq!(
            closer(&[r#"",""#, r#""..""#, r#"":""#, r#""}""#]),
            Some('}')
        );
        // Two closers would be a guess, and no closer means the program simply
        // ran out of input.
        assert_eq!(closer(&[r#"")""#, r#""]""#]), None);
        assert_eq!(closer(&[r#""d""#, r#""(""#]), None);
    }
}
