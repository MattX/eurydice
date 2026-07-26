//! Serializable, frontend-neutral diagnostics produced by the engine.

use lalrpop_util::ParseError;
use miette::SourceSpan;
use serde::Serialize;

use crate::{
    ast::{FunctionDefinition, ParseActionError, Range, Statement, WithRange},
    engine::EngineError,
    error::{PrimitiveArgumentErrorKind, RuntimeError},
    primitives::primitive_signature,
    value::{ElementValue, RuntimeValue},
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

/// A compact value description suitable for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ValueSummary {
    pub shape: String,
    pub outcome_type: String,
    pub collection_size: Option<u32>,
    pub preview: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TraceBinding {
    pub name: String,
    pub value: ValueSummary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EvaluationFrame {
    pub function: String,
    pub call: SourceRange,
    pub definition: Option<SourceRange>,
    pub bindings: Vec<TraceBinding>,
}

/// Stable semantic information for programmatic diagnostic consumers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DiagnosticDetails {
    Syntax,
    UndefinedName {
        name: String,
        namespace: String,
        suggestions: Vec<String>,
    },
    TypeMismatch {
        expected: String,
        actual: Option<ValueSummary>,
    },
    InvalidValue {
        constraint: String,
        actual: String,
    },
    Evaluation,
    Limit {
        setting: String,
        value: usize,
    },
    PoolReuse {
        name: String,
    },
    MissingReturn {
        function: String,
    },
}

/// A complete engine diagnostic. Human-readable strings are included for
/// simple clients; structured fields let richer frontends choose their own UX.
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
    pub details: DiagnosticDetails,
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

pub(crate) fn summarize_value(value: &RuntimeValue) -> ValueSummary {
    let (shape, collection_size) = match value {
        RuntimeValue::Element(ElementValue::Tuple(_)) => ("tuple".to_string(), None),
        RuntimeValue::Element(ElementValue::Enum { .. }) => ("enum".to_string(), None),
        RuntimeValue::Element(_) => ("number".to_string(), None),
        RuntimeValue::List(values, _) => ("sequence".to_string(), u32::try_from(values.len()).ok()),
        RuntimeValue::Pool(pool, _) => ("dice_pool".to_string(), Some(pool.dimension())),
    };
    let preview = value.to_string();
    let preview = if preview.chars().count() > 80 {
        format!("{}…", preview.chars().take(79).collect::<String>())
    } else {
        preview
    };
    ValueSummary {
        shape,
        outcome_type: value.outcome_type().display_name(),
        collection_size,
        preview,
    }
}

fn describe_value(value: &RuntimeValue) -> String {
    match value {
        RuntimeValue::Element(ElementValue::AdditiveIdentity | ElementValue::Int(_)) => {
            "an integer".to_string()
        }
        RuntimeValue::Element(ElementValue::Enum { ty, .. }) => {
            format!("a `{}` value", ty.name)
        }
        RuntimeValue::Element(ElementValue::Tuple(_)) => "a tuple".to_string(),
        RuntimeValue::List(_, outcome_type) => {
            format!("a sequence of `{}` values", outcome_type.display_name())
        }
        RuntimeValue::Pool(_, outcome_type) => {
            format!(
                "a dice pool with `{}` outcomes",
                outcome_type.display_name()
            )
        }
    }
}

fn primitive_call(identifier: &str) -> String {
    primitive_signature(identifier)
        .map(str::to_string)
        .unwrap_or_else(|| format!("[{}]", identifier.replace("{}", "…")))
}

struct DiagnosticParts {
    code: &'static str,
    summary: String,
    labels: Vec<DiagnosticLabel>,
    notes: Vec<String>,
    help: Option<String>,
    fixes: Vec<SuggestedFix>,
    details: DiagnosticDetails,
    incomplete: bool,
}

impl DiagnosticParts {
    fn new(
        code: &'static str,
        summary: impl Into<String>,
        labels: Vec<DiagnosticLabel>,
        details: DiagnosticDetails,
    ) -> Self {
        Self {
            code,
            summary: summary.into(),
            labels,
            notes: Vec::new(),
            help: None,
            fixes: Vec::new(),
            details,
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
            details: self.details,
            incomplete: self.incomplete,
        }
    }

    fn help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }
}

struct ParseDiagnosticParts {
    range: Range,
    diagnostic: DiagnosticParts,
}

impl ParseDiagnosticParts {
    fn new(source: SourceId, range: Range, code: &'static str, summary: impl Into<String>) -> Self {
        Self {
            range,
            diagnostic: DiagnosticParts::new(
                code,
                summary,
                vec![DiagnosticLabel {
                    range: SourceRange { source, range },
                    message: None,
                    style: LabelStyle::Primary,
                }],
                DiagnosticDetails::Syntax,
            ),
        }
    }

    fn incomplete(mut self) -> Self {
        self.diagnostic.incomplete = true;
        self
    }

    fn help(mut self, help: impl Into<String>) -> Self {
        self.diagnostic.help = Some(help.into());
        self
    }

    fn fix(mut self, fix: SuggestedFix) -> Self {
        self.diagnostic.fixes.push(fix);
        self
    }

    fn secondary(mut self, label: DiagnosticLabel) -> Self {
        self.diagnostic.labels.push(label);
        self
    }
}

pub(crate) fn parse_error<T: std::fmt::Display>(
    error: ParseError<usize, T, ParseActionError>,
    source_id: SourceId,
    source: &str,
) -> (EngineError, EngineDiagnostic) {
    let legacy_message = error.to_string();
    let parts = match error {
        ParseError::UnrecognizedToken {
            token, expected, ..
        } => {
            let range = Range::from((token.0, token.2));
            let found = source.get(token.0..token.2).unwrap_or_default();
            if found == "=" && expected.iter().any(|item| item.contains(':')) {
                ParseDiagnosticParts::new(
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
                ParseDiagnosticParts::new(
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
            } else {
                let mut parts = ParseDiagnosticParts::new(
                    source_id,
                    range,
                    "syntax.unexpected_token",
                    format!("Unexpected token `{found}`"),
                );
                parts.diagnostic.help = readable_expected(&expected);
                parts.diagnostic.notes.push(
                    "At the top level, expressions you want to show must start with `output`."
                        .to_string(),
                );
                parts
            }
        }
        ParseError::UnrecognizedEof { location, expected } => {
            let range = Range::from((location, location));
            if let Some((opener, closer)) = unmatched_delimiter(source) {
                ParseDiagnosticParts::new(
                    source_id,
                    range,
                    "syntax.unclosed_delimiter",
                    format!("Missing closing `{closer}`"),
                )
                .incomplete()
                .help(format!("Close the delimiter opened at byte {opener}."))
                .fix(replacement_fix(
                    source_id,
                    range,
                    &closer.to_string(),
                    &format!("Insert `{closer}`"),
                ))
                .secondary(DiagnosticLabel {
                    range: SourceRange {
                        source: source_id,
                        range: Range::from((opener, opener + 1)),
                    },
                    message: Some("opened here".to_string()),
                    style: LabelStyle::Secondary,
                })
            } else {
                let mut parts = ParseDiagnosticParts::new(
                    source_id,
                    range,
                    "syntax.unexpected_end",
                    "The program ends before this construct is complete",
                )
                .incomplete();
                parts.diagnostic.help = readable_expected(&expected);
                parts
            }
        }
        ParseError::ExtraToken { token } => {
            let range = Range::from((token.0, token.2));
            ParseDiagnosticParts::new(
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
                ParseDiagnosticParts::new(
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
                ParseDiagnosticParts::new(
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
                ParseDiagnosticParts::new(
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
            } else if found.is_some_and(|character| character.is_ascii_lowercase()) {
                let end = source[location..]
                    .find(|character: char| !character.is_ascii_lowercase() && character != '_')
                    .map_or(source.len(), |offset| location + offset);
                let range = Range::from((location, end));
                let replacement = source[location..end].to_ascii_uppercase();
                ParseDiagnosticParts::new(
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
                ParseDiagnosticParts::new(
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
            ParseActionError::InvalidIntegerLiteral { range, .. } => ParseDiagnosticParts::new(
                source_id,
                range,
                "syntax.integer_out_of_range",
                "This integer is outside the supported range",
            )
            .help("Integers must fit between -2147483648 and 2147483647."),
            ParseActionError::EmptyFunctionCall { range } => ParseDiagnosticParts::new(
                source_id,
                range,
                "syntax.empty_function_call",
                "A function call cannot be empty",
            )
            .help("Put a lowercase function name between the brackets."),
        },
    };

    let legacy = EngineError::Parse {
        message: legacy_message,
        range: parts.range.into(),
        incomplete: parts.diagnostic.incomplete,
    };
    let diagnostic = parts
        .diagnostic
        .finish(DiagnosticSeverity::Error, Vec::new());
    (legacy, diagnostic)
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
        code: "control_flow.missing_result".to_string(),
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
        details: DiagnosticDetails::MissingReturn {
            function: definition.name.value.clone(),
        },
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

fn unmatched_delimiter(source: &str) -> Option<(usize, char)> {
    let mut stack = Vec::new();
    let mut quoted = false;
    let mut escaped = false;
    for (offset, character) in source.char_indices() {
        if quoted {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                quoted = false;
            }
            continue;
        }
        if character == '"' {
            quoted = true;
            continue;
        }
        match character {
            '(' => stack.push((offset, ')')),
            '[' => stack.push((offset, ']')),
            '{' => stack.push((offset, '}')),
            ')' | ']' | '}' => {
                if stack.last().is_some_and(|(_, close)| *close == character) {
                    stack.pop();
                }
            }
            _ => {}
        }
    }
    stack.pop()
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

fn readable_expected(expected: &[String]) -> Option<String> {
    if expected.is_empty() {
        return None;
    }
    let mut values = expected
        .iter()
        .map(|item| match item.as_str() {
            "r#\"[0-9]+\"#" => "a number".to_string(),
            "r#\"[A-Z_]+\"#" => "an uppercase variable".to_string(),
            other => other.trim_matches('"').to_string(),
        })
        .collect::<Vec<_>>();
    values.sort();
    values.dedup();
    Some(format!("Expected {}.", values.join(", ")))
}

pub(crate) fn runtime_diagnostic(
    error: &RuntimeError,
    source_id: SourceId,
    variable_suggestions: &[String],
    function_suggestions: &[String],
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
    let secondary = |range: &SourceSpan, message: String| DiagnosticLabel {
        range: SourceRange {
            source: source_id,
            range: range.into(),
        },
        message: Some(message),
        style: LabelStyle::Secondary,
    };
    let unlabeled = |range: &SourceSpan, style: LabelStyle| DiagnosticLabel {
        range: SourceRange {
            source: source_id,
            range: range.into(),
        },
        message: None,
        style,
    };

    let mut parts = match error {
        RuntimeError::UndefinedReference { range, name } => DiagnosticParts::new(
            "name.undefined_variable",
            format!("Variable `{name}` is not defined"),
            vec![unlabeled(range, LabelStyle::Primary)],
            DiagnosticDetails::UndefinedName {
                name: name.clone(),
                namespace: "variable".to_string(),
                suggestions: variable_suggestions.to_vec(),
            },
        )
        .help(suggestion_help(
            "Variable names are uppercase and must be assigned before use.",
            variable_suggestions,
        )),
        RuntimeError::UndefinedFunction {
            range, name, help, ..
        } => DiagnosticParts::new(
            "name.undefined_function",
            format!("Function `{name}` is not defined"),
            vec![primary(range, "no matching function signature".to_string())],
            DiagnosticDetails::UndefinedName {
                name: name.clone(),
                namespace: "function".to_string(),
                suggestions: function_suggestions.to_vec(),
            },
        )
        .help(help.clone().unwrap_or_else(|| {
            suggestion_help(
                "Check the function's words and argument positions.",
                function_suggestions,
            )
        })),
        RuntimeError::LoopOverNonSequence { range, found } => DiagnosticParts::new(
            "type.expected_sequence",
            "A loop can only iterate over a sequence",
            vec![primary(range, format!("this evaluates to a {found}"))],
            DiagnosticDetails::TypeMismatch {
                expected: "sequence".to_string(),
                actual: None,
            },
        )
        .help(
            "A dice pool represents a distribution. Pass it through an `s` parameter to \
             evaluate a function once for each possible roll.",
        ),
        RuntimeError::InvalidCondition {
            range,
            found,
            value,
        } => DiagnosticParts::new(
            "type.expected_number",
            "An `if` condition must be a number",
            vec![primary(
                range,
                format!("this is a {found}: {}", summarize_value(value).preview),
            )],
            DiagnosticDetails::TypeMismatch {
                expected: "number".to_string(),
                actual: Some(summarize_value(value)),
            },
        )
        .help(
            "To branch on each possible die result, put the condition in a function with an \
             `n` parameter and pass the die to it.",
        ),
        RuntimeError::InvalidArgumentToOperator {
            operator_range,
            op,
            expected,
            found_range,
            found,
            value,
        } => DiagnosticParts::new(
            "type.operator_argument",
            format!("Operator `{op}` cannot use this value"),
            vec![
                primary(operator_range, format!("`{op}` expects {expected}")),
                secondary(
                    found_range,
                    format!("this is a {found}: {}", summarize_value(value).preview),
                ),
            ],
            DiagnosticDetails::TypeMismatch {
                expected: (*expected).to_string(),
                actual: Some(summarize_value(value)),
            },
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
            DiagnosticDetails::InvalidValue {
                constraint: "nonnegative integer".to_string(),
                actual: value.to_string(),
            },
        ),
        RuntimeError::InvalidPrimitiveArguments(error) => {
            let arguments = &error.arguments;
            let labels = arguments
                .iter()
                .enumerate()
                .map(|(index, argument)| DiagnosticLabel {
                    range: SourceRange {
                        source: source_id,
                        range: (&argument.range).into(),
                    },
                    message: Some(format!(
                        "`{}` is {}: `{}`; expected {}",
                        argument.name,
                        describe_value(&argument.value),
                        summarize_value(&argument.value).preview,
                        argument.expected
                    )),
                    style: if index == 0 {
                        LabelStyle::Primary
                    } else {
                        LabelStyle::Secondary
                    },
                })
                .collect::<Vec<_>>();
            let actual = (arguments.len() == 1).then(|| summarize_value(&arguments[0].value));
            let expected = match arguments.as_slice() {
                [argument] => argument.expected.clone(),
                _ => error.requirement.clone(),
            };
            let code = match error.kind {
                PrimitiveArgumentErrorKind::Type => "type.function_argument",
                PrimitiveArgumentErrorKind::OutcomeType => "type.outcome_mismatch",
            };
            let mut parts = DiagnosticParts::new(
                code,
                format!("`{}` {}", primitive_call(error.function), error.requirement),
                labels,
                DiagnosticDetails::TypeMismatch { expected, actual },
            );
            parts.help = error.help.clone();
            parts
        }
        RuntimeError::InvalidPrimitiveValue(error) => {
            let summary = summarize_value(&error.value);
            let mut parts = DiagnosticParts::new(
                "value.out_of_range",
                format!("`{}` {}", primitive_call(error.function), error.requirement),
                vec![primary(
                    &error.found_range,
                    format!(
                        "`{}` is {}: `{}`; expected {}",
                        error.argument,
                        describe_value(&error.value),
                        summary.preview,
                        error.constraint
                    ),
                )],
                DiagnosticDetails::InvalidValue {
                    constraint: error.constraint.clone(),
                    actual: summary.preview,
                },
            );
            parts.help = error.help.clone();
            parts
        }
        RuntimeError::MathError { range, message } => DiagnosticParts::new(
            "value.arithmetic",
            message,
            vec![unlabeled(range, LabelStyle::Primary)],
            DiagnosticDetails::Evaluation,
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
            DiagnosticDetails::Evaluation,
        ),
        RuntimeError::LabelsOnNonTupleOutput { range } => DiagnosticParts::new(
            "type.labels_require_tuple",
            "Output labels require a tuple-valued output",
            vec![primary(
                range,
                "labels cannot be applied to this value".to_string(),
            )],
            DiagnosticDetails::TypeMismatch {
                expected: "tuple".to_string(),
                actual: None,
            },
        ),
        RuntimeError::OutputLabelCountMismatch {
            range,
            expected,
            found,
        } => DiagnosticParts::new(
            "value.output_label_count",
            format!("Expected {expected} output labels, found {found}"),
            vec![primary(
                range,
                "the label count does not match the tuple".to_string(),
            )],
            DiagnosticDetails::InvalidValue {
                constraint: format!("{expected} labels"),
                actual: found.to_string(),
            },
        ),
        RuntimeError::InvalidRepeatExpression {
            range,
            found,
            value,
        } => DiagnosticParts::new(
            "type.repeat_count",
            "A repetition count must be a number",
            vec![primary(
                range,
                format!("this is a {found}: {}", summarize_value(value).preview),
            )],
            DiagnosticDetails::TypeMismatch {
                expected: "number".to_string(),
                actual: Some(summarize_value(value)),
            },
        ),
        RuntimeError::RangeHasNonSequenceEndpoints { range, found } => DiagnosticParts::new(
            "type.range_endpoint",
            "Both ends of a range must be numbers",
            vec![primary(range, format!("this is a {found}"))],
            DiagnosticDetails::TypeMismatch {
                expected: "number".to_string(),
                actual: None,
            },
        ),
        RuntimeError::EnumTypeError { range, message } => {
            classify_generic_runtime(range, message, &primary)
        }
        RuntimeError::InFunction { .. } => {
            unreachable!("function contexts were peeled before rendering")
        }
    };

    parts
        .labels
        .extend(trace.iter().map(|frame| DiagnosticLabel {
            range: frame.call,
            message: Some(format!(
                "while calling `{}`",
                frame.function.replace("{}", "…")
            )),
            style: LabelStyle::Secondary,
        }));
    parts.finish(DiagnosticSeverity::Error, trace)
}

fn classify_generic_runtime(
    range: &SourceSpan,
    message: &str,
    primary: &impl Fn(&SourceSpan, String) -> DiagnosticLabel,
) -> DiagnosticParts {
    let normalized = message.to_ascii_lowercase();
    let (code, details) = if normalized.contains("out of range")
        || normalized.contains("index must be between")
        || normalized.contains("too many")
    {
        (
            "value.out_of_range",
            DiagnosticDetails::InvalidValue {
                constraint: message.to_string(),
                actual: "out of range".to_string(),
            },
        )
    } else if normalized.contains("operator ") {
        (
            "type.operator_operands",
            DiagnosticDetails::TypeMismatch {
                expected: message.to_string(),
                actual: None,
            },
        )
    } else if normalized.contains("mixed outcome types")
        || normalized.contains("incompatible outcome types")
        || normalized.contains("same outcome type")
        || normalized.starts_with("expected ")
    {
        (
            "type.outcome_mismatch",
            DiagnosticDetails::TypeMismatch {
                expected: message.to_string(),
                actual: None,
            },
        )
    } else if normalized.contains("top level") {
        ("placement.top_level_only", DiagnosticDetails::Evaluation)
    } else if normalized.contains("conflict")
        || normalized.contains("already bound")
        || normalized.contains("already defined")
        || normalized.contains("immutable")
        || normalized.contains("cannot be used as")
    {
        ("name.binding_conflict", DiagnosticDetails::Evaluation)
    } else if normalized.contains("non-additive") || normalized.contains("cannot be summed") {
        (
            "type.non_additive_value",
            DiagnosticDetails::TypeMismatch {
                expected: "an additive value".to_string(),
                actual: None,
            },
        )
    } else {
        ("type.incompatible_value", DiagnosticDetails::Evaluation)
    };

    DiagnosticParts::new(
        code,
        message,
        vec![primary(range, message.to_string())],
        details,
    )
}

fn suggestion_help(base: &str, suggestions: &[String]) -> String {
    match suggestions {
        [] => base.to_string(),
        [only] => format!("{base} Did you mean `{only}`?"),
        many => format!("{base} Similar names: {}.", many.join(", ")),
    }
}

pub(crate) fn undefined_name(error: &RuntimeError) -> Option<&str> {
    match error {
        RuntimeError::UndefinedReference { name, .. }
        | RuntimeError::UndefinedFunction { name, .. } => Some(name),
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
    let Some(name) = undefined_name(error) else {
        return;
    };
    let is_function = matches!(
        peel_runtime_error(error),
        RuntimeError::UndefinedFunction { .. }
    );
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

pub(crate) fn add_adjacent_dice_fix(
    diagnostic: &mut EngineDiagnostic,
    error: &RuntimeError,
    sources: &[DiagnosticSource],
) {
    let RuntimeError::UndefinedFunction {
        help: Some(help), ..
    } = peel_runtime_error(error)
    else {
        return;
    };
    if !help.contains("separate them with a comma") {
        return;
    }
    let Some(primary) = diagnostic.primary_range() else {
        return;
    };
    let Some(source) = sources.iter().find(|source| source.id == primary.source) else {
        return;
    };
    let Some(call) = source.text.get(primary.range.start..primary.range.end) else {
        return;
    };
    let dice_starts = call
        .match_indices(" d")
        .map(|(offset, _)| offset)
        .collect::<Vec<_>>();
    let [_, second, ..] = dice_starts.as_slice() else {
        return;
    };
    let insertion = primary.range.start + second;
    diagnostic.fixes.push(SuggestedFix {
        message: "Separate these dice arguments with a comma".to_string(),
        applicability: FixApplicability::Suggested,
        edits: vec![TextEdit {
            range: SourceRange {
                source: primary.source,
                range: Range {
                    start: insertion,
                    end: insertion,
                },
            },
            replacement: ",".to_string(),
        }],
    });
}

fn peel_runtime_error(error: &RuntimeError) -> &RuntimeError {
    match error {
        RuntimeError::InFunction { source, .. } => peel_runtime_error(source),
        other => other,
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
        DiagnosticDetails::Evaluation,
    )
    .help(help)
}
