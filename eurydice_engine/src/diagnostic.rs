//! Serializable, frontend-neutral diagnostics produced by the engine.

use lalrpop_util::ParseError;
#[cfg(feature = "serde")]
use serde::Serialize;

use crate::{
    ast::{ByteRange, FunctionDefinition, ParseActionError, Statement, WithRange},
    error::{ArityMismatch, NonAdditiveSubject, RuntimeError},
    primitives::primitive_signature,
    value::{ElementValue, RuntimeValue, SymbolTable},
};

/// Identifies one source submission within a stateful [`crate::Engine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct SourceId(pub u64);

/// A block of source text referenced by a diagnostic.
///
/// A new `DiagnosticSource` is created each time a block of text is submitted to
/// an [`crate::Engine`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[non_exhaustive]
pub struct DiagnosticSource {
    /// A unique identifier for the source.
    pub id: SourceId,
    /// The source's user-defined name.
    pub name: String,
    /// The source's full text.
    pub text: String,
}

/// A range whose source submission is explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct SourceRange {
    /// The unique source identifier assigned by the engine.
    pub source: SourceId,
    /// The byte range within this source.
    pub range: ByteRange,
}

/// Describes a diagnostic's severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize), serde(rename_all = "snake_case"))]
#[non_exhaustive]
pub enum DiagnosticSeverity {
    /// Used when a diagnostic caused a compilation or execution failure.
    Error,
    /// Used for diagnostics that did not cause a compilation or execution failure,
    /// but may provide useful information to the user.
    Warning,
}

/// A span of source text that is relevant to a diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[non_exhaustive]
pub struct DiagnosticLabel {
    /// The range of source text
    pub range: SourceRange,
    /// An optional explanation for the span's relevance to the diagnostic.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct TextEdit {
    pub range: SourceRange,
    pub replacement: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[non_exhaustive]
pub struct SuggestedFix {
    pub message: String,
    pub edits: Vec<TextEdit>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct TraceBinding {
    pub name: String,
    /// The bound value, rendered for display.
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[non_exhaustive]
pub struct EvaluationFrame {
    pub function: String,
    pub call: SourceRange,
    pub definition: Option<SourceRange>,
    pub bindings: Vec<TraceBinding>,
}

/// What kind of problem a diagnostic reports.
///
/// Codes are the stable part of a diagnostic: summaries and help text
/// are prose that may be reworded at any time, but a code means the same thing
/// across versions, and serializes to the dotted string it is named for.
/// Consumers that branch on a diagnostic should branch on this.
///
/// The set grows as the engine learns to report more, so it is
/// `#[non_exhaustive]`: a match on it needs a fallback arm, and a new code is
/// not a breaking change. [`DiagnosticCode::ALL`] lists every code this version
/// defines. The declaration table below generates that list and
/// [`as_str`](DiagnosticCode::as_str), [`severity`](DiagnosticCode::severity),
/// and [`is_incomplete`](DiagnosticCode::is_incomplete), keeping them exhaustive
/// by construction.
macro_rules! define_diagnostic_codes {
    ($( $(#[$metadata:meta])* $variant:ident => ($wire:literal, $severity:ident, $incomplete:literal), )+) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum DiagnosticCode {
            $( $(#[$metadata])* $variant, )+
        }

        impl DiagnosticCode {
            /// Every code this version of the engine can produce.
            pub const ALL: &'static [DiagnosticCode] = &[
                $(Self::$variant,)+
            ];

            /// The code's stable dotted string, as it is serialized.
            #[must_use]
            pub fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $wire,)+
                }
            }

            /// Whether this diagnostic prevents the submission from running.
            #[must_use]
            pub const fn severity(self) -> DiagnosticSeverity {
                match self {
                    $(Self::$variant => DiagnosticSeverity::$severity,)+
                }
            }

            /// Whether more input may complete a submission that ended with this diagnostic.
            #[must_use]
            pub const fn is_incomplete(self) -> bool {
                match self {
                    $(Self::$variant => $incomplete,)+
                }
            }
        }
    };
}

define_diagnostic_codes! {
    /// `=` was used where a binding expects `:`.
    AssignmentSeparator => ("syntax.assignment_separator", Error, false),
    /// `==` was used where a comparison expects `=`.
    EqualityOperator => ("syntax.equality_operator", Error, false),
    /// A variable was written in lowercase, where names are uppercase.
    VariableCase => ("syntax.variable_case", Error, false),
    /// A token cannot appear where it was found.
    UnexpectedToken => ("syntax.unexpected_token", Error, false),
    /// A bracket, brace, or parenthesis was never closed.
    UnclosedDelimiter => ("syntax.unclosed_delimiter", Error, true),
    /// The program ended in the middle of something.
    UnexpectedEnd => ("syntax.unexpected_end", Error, true),
    /// Input continues past the end of a complete program.
    ExtraToken => ("syntax.extra_token", Error, false),
    /// A string literal was never closed.
    UnterminatedString => ("syntax.unterminated_string", Error, true),
    /// A block comment was never closed.
    UnterminatedComment => ("syntax.unterminated_comment", Error, true),
    /// A `;` was used to end a statement, which the language does not use.
    UnexpectedSemicolon => ("syntax.unexpected_semicolon", Error, false),
    /// A character that means nothing in the language.
    InvalidCharacter => ("syntax.invalid_character", Error, false),
    /// An integer literal does not fit in the engine's integer type.
    IntegerOutOfRange => ("syntax.integer_out_of_range", Error, false),
    /// A call names no function at all, as in `[]`.
    EmptyFunctionCall => ("syntax.empty_function_call", Error, false),
    /// A variable that has not been assigned.
    UndefinedVariable => ("name.undefined_variable", Error, false),
    /// A function that has not been defined.
    UndefinedFunction => ("name.undefined_function", Error, false),
    /// A name is already taken, or cannot be bound in this position.
    BindingConflict => ("name.binding_conflict", Error, false),
    /// A sequence was required.
    ExpectedSequence => ("type.expected_sequence", Error, false),
    /// A number was required.
    ExpectedNumber => ("type.expected_number", Error, false),
    /// An operator was given an operand of a type it does not accept.
    OperatorArgument => ("type.operator_argument", Error, false),
    /// An operator's operands do not work together, whatever each is alone.
    OperatorOperands => ("type.operator_operands", Error, false),
    /// A function was given an argument of a type it does not accept.
    FunctionArgument => ("type.function_argument", Error, false),
    /// Two values that had to line up have different shapes.
    OutcomeMismatch => ("type.outcome_mismatch", Error, false),
    /// A value that had to be summed has parts that cannot be added.
    NonAdditiveValue => ("type.non_additive_value", Error, false),
    /// `labeled` was applied to an output that is not a tuple.
    LabelsRequireTuple => ("type.labels_require_tuple", Error, false),
    /// A repeat count is not a number.
    RepeatCount => ("type.repeat_count", Error, false),
    /// A range endpoint is not a number.
    RangeEndpoint => ("type.range_endpoint", Error, false),
    /// A negative number was given where only zero or more makes sense.
    NonnegativeRequired => ("value.nonnegative_required", Error, false),
    /// A value falls outside the range the operation allows.
    OutOfRange => ("value.out_of_range", Error, false),
    /// An arithmetic operation has no defined result, such as division by zero.
    ArithmeticError => ("value.arithmetic_error", Error, false),
    /// The number of labels does not match the number of tuple fields.
    OutputLabelCount => ("value.output_label_count", Error, false),
    /// `result:` appeared outside a function.
    ResultOutsideFunction => ("placement.result_outside_function", Error, false),
    /// A statement that is only allowed at the top level appeared inside a block.
    TopLevelOnly => ("placement.top_level_only", Error, false),
    /// A function may finish without reaching `result:`.
    MissingResult => ("control_flow.missing_result", Warning, false),
    /// One dice pool is sampled independently more than once.
    IndependentPoolReuse => ("evaluation.independent_pool_reuse", Warning, false),
    /// Exploding dice stopped at the configured depth.
    ExplodeDepth => ("evaluation.explode_depth", Warning, false),
    /// Recursion stopped at the configured depth.
    MaximumFunctionDepth => ("evaluation.maximum_function_depth", Warning, false),
}

impl std::fmt::Display for DiagnosticCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Serialized as its dotted string, never as a variant name, so the wire
/// contract is the one [`as_str`](DiagnosticCode::as_str) documents and cannot
/// drift from it.
#[cfg(feature = "serde")]
impl Serialize for DiagnosticCode {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

/// Errors, warnings, or other information produced by the engine about a span of code.
///
/// The `code` classifies the diagnostic for consumers that want to branch on
/// it. Other fields provide additional context to an end user.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[non_exhaustive]
pub struct Diagnostic {
    /// The error type.
    pub code: DiagnosticCode,
    /// A human-readable description of the summary.
    pub summary: String,
    /// The code span that is the primary cause of this diagnostic.
    pub primary_label: DiagnosticLabel,
    /// Code spans that provide additional context for this diagnostic.
    pub secondary_labels: Vec<DiagnosticLabel>,
    /// Optional human-readable context or description of how to fix the diagnostic.
    pub help: Option<String>,
    /// An optional fix for the diagnostic.
    pub fix: Option<SuggestedFix>,
    /// Stack trace when this diagnostic was emitted. Empty for syntax issues.
    pub trace: Vec<EvaluationFrame>,
}

impl Diagnostic {
    /// Every label associated with this diagnostic, primary first.
    pub fn labels(&self) -> impl Iterator<Item = &DiagnosticLabel> {
        std::iter::once(&self.primary_label).chain(&self.secondary_labels)
    }

    /// Every label associated with this diagnostic, primary first.
    pub fn labels_mut(&mut self) -> impl Iterator<Item = &mut DiagnosticLabel> {
        std::iter::once(&mut self.primary_label).chain(&mut self.secondary_labels)
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

fn describe_value(value: &RuntimeValue) -> String {
    match value {
        RuntimeValue::Element(ElementValue::AdditiveIdentity | ElementValue::Int(_)) => {
            "an integer".to_string()
        }
        RuntimeValue::Element(ElementValue::Symbol(_)) => "a symbol".to_string(),
        RuntimeValue::Element(ElementValue::Tuple(_)) => "a tuple".to_string(),
        RuntimeValue::List(values) => format!(
            "a sequence of `{}` values",
            describe_outcomes(values.iter())
        ),
        RuntimeValue::Pool(pool) => format!(
            "a dice pool with `{}` outcomes",
            describe_outcomes(pool.ordered_outcomes().iter().map(|(outcome, _)| outcome))
        ),
    }
}

/// Names the kinds of value a collection holds, read off the values.
///
/// Nothing records this anywhere, so a collection with no values has nothing to
/// say about itself and reads as empty.
fn describe_outcomes<'a>(values: impl IntoIterator<Item = &'a ElementValue>) -> String {
    let mut kinds = Vec::new();
    let push = |kinds: &mut Vec<&str>, kind: &'static str| {
        if !kinds.contains(&kind) {
            kinds.push(kind);
        }
    };
    for value in values {
        match value {
            ElementValue::Symbol(_) => push(&mut kinds, "symbol"),
            ElementValue::Tuple(_) => push(&mut kinds, "tuple"),
            ElementValue::Int(_) | ElementValue::AdditiveIdentity => push(&mut kinds, "int"),
        }
    }
    // Ints lead, so a mostly-numeric collection reads the way it did before
    // symbols existed.
    kinds.sort_by_key(|kind| *kind != "int");
    match kinds.len() {
        0 => "empty".to_string(),
        _ => kinds.join(" or "),
    }
}

/// How a value is shaped, which is the only thing two values an operation
/// combines can irreconcilably disagree about.
fn shape_name(value: &ElementValue) -> String {
    match value {
        ElementValue::Tuple(fields) => format!("a tuple of {} fields", fields.len()),
        ElementValue::AdditiveIdentity => "the empty sum".to_string(),
        ElementValue::Int(_) | ElementValue::Symbol(_) => "a single value".to_string(),
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
    range: &ByteRange,
    expected: &str,
    value: &RuntimeValue,
    symbols: &SymbolTable,
) -> DiagnosticLabel {
    DiagnosticLabel {
        range: SourceRange {
            source,
            range: *range,
        },
        message: Some(format!(
            "`{}` is {}; expected {expected}",
            preview_value(value, symbols),
            describe_value(value)
        )),
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
        describe_value(value),
        preview_value(value, symbols)
    )
}

struct DiagnosticParts {
    code: DiagnosticCode,
    summary: String,
    primary_label: DiagnosticLabel,
    secondary_labels: Vec<DiagnosticLabel>,
    help: Option<String>,
    fix: Option<SuggestedFix>,
}

impl DiagnosticParts {
    fn new(
        code: DiagnosticCode,
        summary: impl Into<String>,
        primary_label: DiagnosticLabel,
    ) -> Self {
        Self {
            code,
            summary: summary.into(),
            primary_label,
            secondary_labels: Vec::new(),
            help: None,
            fix: None,
        }
    }

    fn finish(self, trace: Vec<EvaluationFrame>) -> Diagnostic {
        Diagnostic {
            code: self.code,
            summary: self.summary,
            primary_label: self.primary_label,
            secondary_labels: self.secondary_labels,
            help: self.help,
            fix: self.fix,
            trace,
        }
    }

    fn secondary_labels(mut self, labels: Vec<DiagnosticLabel>) -> Self {
        self.secondary_labels = labels;
        self
    }

    fn help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    fn maybe_help(mut self, help: Option<String>) -> Self {
        self.help = help;
        self
    }

    fn fix(mut self, fix: SuggestedFix) -> Self {
        self.fix = Some(fix);
        self
    }
}

/// A syntax diagnostic with a single unlabeled primary span.
fn syntax_parts(
    source: SourceId,
    range: ByteRange,
    code: DiagnosticCode,
    summary: impl Into<String>,
) -> DiagnosticParts {
    DiagnosticParts::new(
        code,
        summary,
        DiagnosticLabel {
            range: SourceRange { source, range },
            message: None,
        },
    )
}

pub(crate) fn parse_error<T: std::fmt::Display>(
    error: ParseError<usize, T, ParseActionError>,
    source_id: SourceId,
    source: &str,
) -> Diagnostic {
    let parts = match error {
        ParseError::UnrecognizedToken {
            token, expected, ..
        } => {
            let range = ByteRange::from((token.0, token.2));
            let found = source.get(token.0..token.2).unwrap_or_default();
            if found == "=" && expected.iter().any(|item| item.contains(':')) {
                syntax_parts(
                    source_id,
                    range,
                    DiagnosticCode::AssignmentSeparator,
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
                    DiagnosticCode::EqualityOperator,
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
                    DiagnosticCode::VariableCase,
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
                    DiagnosticCode::UnexpectedToken,
                    format!("Unexpected token `{found}`"),
                )
                .help("At the top level, expressions you want to show must start with `output`.")
            }
        }
        ParseError::UnrecognizedEof { location, expected } => {
            let range = ByteRange::from((location, location));
            if let Some(closer) = expected_closer(&expected) {
                syntax_parts(
                    source_id,
                    range,
                    DiagnosticCode::UnclosedDelimiter,
                    format!("Missing closing `{closer}`"),
                )
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
                    DiagnosticCode::UnexpectedEnd,
                    "The program ends before this construct is complete",
                )
            }
        }
        ParseError::ExtraToken { token } => {
            let range = ByteRange::from((token.0, token.2));
            syntax_parts(
                source_id,
                range,
                DiagnosticCode::ExtraToken,
                format!(
                    "Unexpected extra token `{}`",
                    source.get(token.0..token.2).unwrap_or_default()
                ),
            )
        }
        ParseError::InvalidToken { location } => {
            let (end, found) = next_char(source, location);
            let range = ByteRange::from((location, end));
            let insertion = ByteRange {
                start: source.len(),
                end: source.len(),
            };
            if found == Some('"') && !has_unescaped_quote(&source[end..]) {
                syntax_parts(
                    source_id,
                    range,
                    DiagnosticCode::UnterminatedString,
                    "This string is missing its closing quote",
                )
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
                    DiagnosticCode::UnterminatedComment,
                    "This block comment is missing its closing backslash",
                )
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
                    DiagnosticCode::UnexpectedSemicolon,
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
                    DiagnosticCode::InvalidCharacter,
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
                DiagnosticCode::IntegerOutOfRange,
                "This integer is outside the supported range",
            )
            .help("Integers must fit between -2147483648 and 2147483647."),
            ParseActionError::EmptyFunctionCall { range } => syntax_parts(
                source_id,
                range,
                DiagnosticCode::EmptyFunctionCall,
                "A function call cannot be empty",
            )
            .help("Put a lowercase function name between the brackets."),
        },
    };

    parts.finish(Vec::new())
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

pub(crate) fn missing_return_warning(
    source: SourceId,
    definition: &FunctionDefinition,
) -> Option<Diagnostic> {
    if block_always_returns(&definition.body) {
        return None;
    }

    let function = definition.name.value.replace("{}", "…");
    Some(Diagnostic {
        code: DiagnosticCode::MissingResult,
        summary: format!("Function `[{function}]` may finish without a result"),
        primary_label: DiagnosticLabel {
            range: SourceRange {
                source,
                range: definition.name.range,
            },
            message: Some("not every path reaches `result:`".to_string()),
        },
        secondary_labels: Vec::new(),
        help: Some(
            "When execution reaches the end of a function, the function produces an empty die. \
             Add `result:` on every path through this function."
                .to_string(),
        ),
        fix: None,
        trace: Vec::new(),
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
    range: ByteRange,
    replacement: &str,
    message: &str,
) -> SuggestedFix {
    SuggestedFix {
        message: message.to_string(),
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
) -> Diagnostic {
    let mut trace = Vec::new();
    let (error, source_id) = runtime_context(error, source_id, &mut trace);
    let primary = |range: &ByteRange, message: String| DiagnosticLabel {
        range: SourceRange {
            source: source_id,
            range: *range,
        },
        message: Some(message),
    };
    let unlabeled = |range: &ByteRange| DiagnosticLabel {
        range: SourceRange {
            source: source_id,
            range: *range,
        },
        message: None,
    };

    let parts = match error {
        RuntimeError::UndefinedReference { range, name } => DiagnosticParts::new(
            DiagnosticCode::UndefinedVariable,
            format!("Variable `{name}` is not defined"),
            unlabeled(range),
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
                DiagnosticCode::UndefinedFunction,
                format!("Function `[{}]` is not defined", name.replace("{}", "…")),
                unlabeled(range),
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
                    edits: vec![TextEdit {
                        range: SourceRange {
                            source: source_id,
                            range: ByteRange {
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
                DiagnosticCode::ExpectedSequence,
                "A loop can only iterate over a sequence",
                type_mismatch_label(source_id, range, "a sequence", value, symbols),
            );
            if matches!(value, RuntimeValue::Pool(_)) {
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
                DiagnosticCode::ExpectedNumber,
                "An `if` condition must be an integer",
                type_mismatch_label(source_id, range, "an integer", value, symbols),
            );
            if matches!(value, RuntimeValue::Pool(_)) {
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
            DiagnosticCode::OperatorArgument,
            format!("Operator `{op}` expects {expected}"),
            primary(
                found_range,
                format!(
                    "`{}` is {}",
                    preview_value(value, symbols),
                    describe_value(value)
                ),
            ),
        )
        .secondary_labels(vec![unlabeled(operator_range)]),
        RuntimeError::NegativeArgumentToFunction {
            range,
            name,
            found_range,
            value,
        } => DiagnosticParts::new(
            DiagnosticCode::NonnegativeRequired,
            format!("`{}` needs a nonnegative count", primitive_call(name)),
            primary(found_range, format!("this evaluates to {value}")),
        )
        .secondary_labels(vec![unlabeled(range)]),
        RuntimeError::InvalidPrimitiveArguments(error) => {
            let mut labels = error.arguments.iter().map(|argument| DiagnosticLabel {
                range: SourceRange {
                    source: source_id,
                    range: argument.range,
                },
                message: Some(argument_mismatch_message(
                    argument.name,
                    &argument.value,
                    &argument.expected,
                    symbols,
                )),
            });
            let primary_label = labels
                .next()
                .expect("an invalid primitive argument report has an invalid argument");
            DiagnosticParts::new(
                DiagnosticCode::FunctionArgument,
                format!("`{}` {}", primitive_call(error.function), error.requirement),
                primary_label,
            )
            .secondary_labels(labels.collect())
            .maybe_help(error.help.clone())
        }
        RuntimeError::InvalidPrimitiveValue(error) => DiagnosticParts::new(
            DiagnosticCode::OutOfRange,
            format!("`{}` {}", primitive_call(error.function), error.requirement),
            primary(
                &error.found_range,
                argument_mismatch_message(error.argument, &error.value, &error.constraint, symbols),
            ),
        )
        .maybe_help(error.help.clone()),
        RuntimeError::ShapeMismatch(error) => {
            let first = shape_name(&error.first);
            let second = shape_name(&error.second);
            let mut summary = format!("{first} and {second} cannot be combined");
            summary[..1].make_ascii_uppercase();
            DiagnosticParts::new(
                DiagnosticCode::OutcomeMismatch,
                summary,
                primary(
                    &error.range,
                    format!(
                        "{} needs them to line up, but `{}` is {first} and `{}` is {second}",
                        error.action,
                        error.first.display(symbols),
                        error.second.display(symbols)
                    ),
                ),
            )
            .help(
                "Values may be numbers or symbols and still share a sequence or a pool, but an \
                 operation that combines them needs them to be all single values, or all tuples \
                 of the same size.",
            )
        }
        RuntimeError::NonAdditiveSum(error) => {
            let (subject, parts_of_it) = match error.subject {
                NonAdditiveSubject::Pool(dimension) => {
                    (Some(format!("a pool of {dimension} dice")), "outcomes")
                }
                NonAdditiveSubject::Sequence(length) => {
                    (Some(format!("a sequence of {length} values")), "values")
                }
                // No collection is involved, so the witness *is* the subject
                // and the label alone carries it.
                NonAdditiveSubject::Operand => (None, "values"),
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
            let summary = match subject {
                Some(subject) => format!(
                    "{} requires summing {subject}, but they cannot be added together",
                    error.action
                ),
                None => format!("{} requires numbers", error.action),
            };
            DiagnosticParts::new(
                DiagnosticCode::NonAdditiveValue,
                summary,
                primary(&error.range, label),
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
        RuntimeError::MathError { range, message } => {
            DiagnosticParts::new(DiagnosticCode::ArithmeticError, message, unlabeled(range))
        }
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
            DiagnosticCode::ResultOutsideFunction,
            "`result:` can only appear inside a function",
            unlabeled(range),
        ),
        RuntimeError::LabelsOnNonTupleOutput {
            range,
            value_range,
            value,
        } => DiagnosticParts::new(
            DiagnosticCode::LabelsRequireTuple,
            "Output labels require a tuple-valued output",
            type_mismatch_label(source_id, value_range, "a tuple", value, symbols),
        )
        .secondary_labels(vec![unlabeled(range)]),
        RuntimeError::OutputLabelCountMismatch {
            range,
            expected,
            found,
        } => DiagnosticParts::new(
            DiagnosticCode::OutputLabelCount,
            format!(
                "This tuple has {expected} fields, but {found} output {} provided",
                if *found == 1 {
                    "label was"
                } else {
                    "labels were"
                }
            ),
            unlabeled(range),
        ),
        RuntimeError::InvalidRepeatExpression { range, value } => DiagnosticParts::new(
            DiagnosticCode::RepeatCount,
            "A repetition count must be an integer",
            type_mismatch_label(source_id, range, "an integer", value, symbols),
        ),
        RuntimeError::RangeHasNonSequenceEndpoints { range, value } => DiagnosticParts::new(
            DiagnosticCode::RangeEndpoint,
            "Both ends of a range must be integers",
            type_mismatch_label(source_id, range, "an integer", value, symbols),
        ),
        RuntimeError::Semantic {
            code,
            range,
            message,
        } => DiagnosticParts::new(*code, message, unlabeled(range)),
        RuntimeError::InFunction { .. } => {
            unreachable!("function contexts were peeled before rendering")
        }
    };

    parts.finish(trace)
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
    diagnostic: &mut Diagnostic,
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
        diagnostic.secondary_labels.push(DiagnosticLabel {
            range: SourceRange { source, range },
            message: Some("defined later".to_string()),
        });
        let context = "Statements execute in order, so this definition is not available yet.";
        match &mut diagnostic.help {
            Some(help) => {
                help.push(' ');
                help.push_str(context);
            }
            None => diagnostic.help = Some(context.to_string()),
        }
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
    range: &ByteRange,
    source_id: SourceId,
    help: &str,
) -> DiagnosticParts {
    DiagnosticParts::new(
        DiagnosticCode::TopLevelOnly,
        format!("`{statement}` can only appear at the top level"),
        DiagnosticLabel {
            range: SourceRange {
                source: source_id,
                range: *range,
            },
            message: None,
        },
    )
    .help(help)
}

#[cfg(test)]
mod tests {
    use super::{DiagnosticCode, expected_closer};

    /// The strings are the diagnostic contract a frontend matches on, so they
    /// are pinned here rather than left to follow whatever a variant happens to
    /// be renamed to. A new code belongs at the end of this list.
    #[test]
    fn every_code_has_a_stable_string() {
        let strings = DiagnosticCode::ALL
            .iter()
            .map(|code| code.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            strings,
            [
                "syntax.assignment_separator",
                "syntax.equality_operator",
                "syntax.variable_case",
                "syntax.unexpected_token",
                "syntax.unclosed_delimiter",
                "syntax.unexpected_end",
                "syntax.extra_token",
                "syntax.unterminated_string",
                "syntax.unterminated_comment",
                "syntax.unexpected_semicolon",
                "syntax.invalid_character",
                "syntax.integer_out_of_range",
                "syntax.empty_function_call",
                "name.undefined_variable",
                "name.undefined_function",
                "name.binding_conflict",
                "type.expected_sequence",
                "type.expected_number",
                "type.operator_argument",
                "type.operator_operands",
                "type.function_argument",
                "type.outcome_mismatch",
                "type.non_additive_value",
                "type.labels_require_tuple",
                "type.repeat_count",
                "type.range_endpoint",
                "value.nonnegative_required",
                "value.out_of_range",
                "value.arithmetic_error",
                "value.output_label_count",
                "placement.result_outside_function",
                "placement.top_level_only",
                "control_flow.missing_result",
                "evaluation.independent_pool_reuse",
                "evaluation.explode_depth",
                "evaluation.maximum_function_depth",
            ]
        );
    }

    /// Frontends read the code as a plain string. Serializing it as anything
    /// structured — a tagged variant, an object — would break every one of
    /// them, and nothing about the enum itself would look wrong.
    #[cfg(feature = "serde")]
    #[test]
    fn codes_serialize_as_a_plain_string() {
        for code in DiagnosticCode::ALL {
            assert_eq!(
                serde_lexpr::to_string(code).unwrap(),
                format!("\"{}\"", code.as_str())
            );
        }
    }

    /// A code that serialized to the same string as another would make the two
    /// indistinguishable to every frontend.
    #[test]
    fn codes_are_distinct() {
        let mut seen = std::collections::HashSet::new();
        for code in DiagnosticCode::ALL {
            assert!(seen.insert(code.as_str()), "duplicate code {code}");
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
