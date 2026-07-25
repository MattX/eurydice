//! This contains graphical utilities used both by the CLI and the integ tests.
use std::collections::HashSet;

use eurydice_engine::{
    DiagnosticSeverity, DiagnosticSource, EngineDiagnostic, LabelStyle,
    output::{Distribution, FieldSchema},
};
use miette::{Diagnostic, GraphicalReportHandler, LabeledSpan, NamedSource, Severity};

pub fn format_output_probabilities(distribution: Distribution) -> Vec<(String, f64)> {
    let is_tuple = distribution.fields.len() > 1;
    distribution
        .probabilities
        .into_iter()
        .map(|(outcome, probability)| {
            let fields = outcome
                .iter()
                .zip(&distribution.fields)
                .enumerate()
                .map(|(index, (value, schema))| {
                    let value = match schema {
                        FieldSchema::Int => value.to_string(),
                        FieldSchema::Enum { labels, .. } => usize::try_from(*value)
                            .ok()
                            .and_then(|i| labels.get(i))
                            .cloned()
                            .unwrap_or_else(|| value.to_string()),
                    };
                    distribution
                        .field_names
                        .as_ref()
                        .and_then(|names| names.get(index))
                        .map_or(value.clone(), |name| format!("{name}: {value}"))
                })
                .collect::<Vec<_>>();
            let label = if is_tuple {
                format!("({})", fields.join(", "))
            } else {
                fields.into_iter().next().unwrap_or_default()
            };
            (label, probability)
        })
        .collect()
}

pub fn print_diagnostic<T>(diagnostic: T, source_code: &str)
where
    T: Diagnostic,
{
    let handler = GraphicalReportHandler::new();
    let mut string = String::new();
    let wrapped = DiagnosticSourceAdapter {
        source: diagnostic,
        source_code,
    };
    let _ = handler.render_report(&mut string, &wrapped);
    eprintln!("{}", string);
}

/// Renders the engine's frontend-neutral diagnostics for terminal users.
pub fn format_engine_diagnostics(
    diagnostics: &[EngineDiagnostic],
    sources: &[DiagnosticSource],
) -> String {
    let mut rendered = String::new();
    for diagnostic in diagnostics {
        let source_ids = diagnostic
            .labels
            .iter()
            .map(|label| label.range.source)
            .collect::<HashSet<_>>();
        let mut first_source = true;
        for source in sources
            .iter()
            .filter(|source| source_ids.contains(&source.id))
        {
            let labels = diagnostic
                .labels
                .iter()
                .filter(|label| label.range.source == source.id)
                .map(|label| {
                    let span = (
                        label.range.range.start,
                        label.range.range.end - label.range.range.start,
                    );
                    if label.style == LabelStyle::Primary {
                        LabeledSpan::new_primary_with_span(Some(label.message.clone()), span)
                    } else {
                        LabeledSpan::new_with_span(Some(label.message.clone()), span)
                    }
                })
                .collect();
            let adapter = EngineDiagnosticAdapter {
                summary: diagnostic.summary.clone(),
                code: diagnostic.code.clone(),
                severity: match diagnostic.severity {
                    DiagnosticSeverity::Error => Severity::Error,
                    DiagnosticSeverity::Warning => Severity::Warning,
                },
                help: first_source.then(|| diagnostic.help.clone()).flatten(),
                source_code: NamedSource::new(source.name.clone(), source.text.clone()),
                labels,
            };
            let _ = GraphicalReportHandler::new().render_report(&mut rendered, &adapter);
            first_source = false;
        }

        for note in &diagnostic.notes {
            rendered.push_str(&format!("  note: {note}\n"));
        }
        for fix in &diagnostic.fixes {
            rendered.push_str(&format!("  suggestion: {}\n", fix.message));
        }
        for frame in &diagnostic.trace {
            let source = sources.iter().find(|source| source.id == frame.call.source);
            let location = source.map_or_else(
                || format!("submission {}", frame.call.source.0 + 1),
                |source| {
                    let (line, column) = line_column(&source.text, frame.call.range.start);
                    format!("{}:{line}:{column}", source.name)
                },
            );
            let bindings = frame
                .bindings
                .iter()
                .map(|binding| format!("{} = {}", binding.name, binding.value.preview))
                .collect::<Vec<_>>()
                .join(", ");
            let bindings = if bindings.is_empty() {
                String::new()
            } else {
                format!(" with {bindings}")
            };
            rendered.push_str(&format!(
                "  while calling [{}] at {location}{bindings}\n",
                frame.function.replace("{}", "…")
            ));
        }
    }
    rendered
}

pub fn print_engine_diagnostics(diagnostics: &[EngineDiagnostic], sources: &[DiagnosticSource]) {
    eprint!("{}", format_engine_diagnostics(diagnostics, sources));
}

fn line_column(source: &str, byte_offset: usize) -> (usize, usize) {
    let prefix = source.get(..byte_offset).unwrap_or(source);
    let line = prefix.bytes().filter(|byte| *byte == b'\n').count() + 1;
    let column = prefix
        .rsplit('\n')
        .next()
        .map_or(1, |line| line.chars().count() + 1);
    (line, column)
}

struct EngineDiagnosticAdapter {
    summary: String,
    code: String,
    severity: Severity,
    help: Option<String>,
    source_code: NamedSource<String>,
    labels: Vec<LabeledSpan>,
}

impl std::fmt::Display for EngineDiagnosticAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.summary)
    }
}

impl std::fmt::Debug for EngineDiagnosticAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineDiagnosticAdapter")
            .field("code", &self.code)
            .field("summary", &self.summary)
            .finish()
    }
}

impl std::error::Error for EngineDiagnosticAdapter {}

impl Diagnostic for EngineDiagnosticAdapter {
    fn source_code(&self) -> Option<&dyn miette::SourceCode> {
        Some(&self.source_code)
    }

    fn code<'a>(&'a self) -> Option<Box<dyn std::fmt::Display + 'a>> {
        Some(Box::new(&self.code))
    }

    fn severity(&self) -> Option<Severity> {
        Some(self.severity)
    }

    fn help<'a>(&'a self) -> Option<Box<dyn std::fmt::Display + 'a>> {
        self.help
            .as_ref()
            .map(|help| Box::new(help) as Box<dyn std::fmt::Display>)
    }

    fn labels(&self) -> Option<Box<dyn Iterator<Item = LabeledSpan> + '_>> {
        Some(Box::new(self.labels.iter().cloned()))
    }
}

struct DiagnosticSourceAdapter<T, U: miette::SourceCode>
where
    T: std::error::Error,
    T: Diagnostic,
{
    source: T,
    source_code: U,
}

impl<T, U> std::fmt::Display for DiagnosticSourceAdapter<T, U>
where
    T: std::error::Error,
    T: Diagnostic,
    U: miette::SourceCode,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.source)
    }
}

impl<T, U> std::fmt::Debug for DiagnosticSourceAdapter<T, U>
where
    T: std::error::Error,
    T: Diagnostic,
    U: miette::SourceCode,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.source)
    }
}

impl<T, U> std::error::Error for DiagnosticSourceAdapter<T, U>
where
    T: std::error::Error,
    T: Diagnostic,
    U: miette::SourceCode,
{
}

impl<T, U> Diagnostic for DiagnosticSourceAdapter<T, U>
where
    T: std::error::Error,
    T: Diagnostic,
    U: miette::SourceCode,
{
    fn source_code(&self) -> Option<&dyn miette::SourceCode> {
        Some(&self.source_code)
    }

    fn code<'a>(&'a self) -> Option<Box<dyn std::fmt::Display + 'a>> {
        self.source.code()
    }

    fn severity(&self) -> Option<miette::Severity> {
        self.source.severity()
    }

    fn help<'a>(&'a self) -> Option<Box<dyn std::fmt::Display + 'a>> {
        self.source.help()
    }

    fn url<'a>(&'a self) -> Option<Box<dyn std::fmt::Display + 'a>> {
        self.source.url()
    }

    fn labels(&self) -> Option<Box<dyn Iterator<Item = miette::LabeledSpan> + '_>> {
        self.source.labels()
    }

    fn related<'a>(&'a self) -> Option<Box<dyn Iterator<Item = &'a dyn Diagnostic> + 'a>> {
        self.source.related()
    }

    fn diagnostic_source(&self) -> Option<&dyn Diagnostic> {
        self.source.diagnostic_source()
    }
}

#[cfg(test)]
mod tests {
    use super::{format_engine_diagnostics, format_output_probabilities};
    use eurydice_engine::Engine;
    use eurydice_engine::output::{Distribution, FieldSchema};

    #[test]
    fn formats_labeled_and_unlabeled_tuple_outcomes() {
        let output = |field_names| Distribution {
            fields: vec![FieldSchema::Int, FieldSchema::Int],
            field_names,
            probabilities: vec![(vec![1, 2], 1.0)],
        };

        assert_eq!(format_output_probabilities(output(None))[0].0, "(1, 2)");
        assert_eq!(
            format_output_probabilities(output(Some(vec!["A".into(), "B".into()])))[0].0,
            "(A: 1, B: 2)"
        );

        let enum_output = Distribution {
            fields: vec![
                FieldSchema::Int,
                FieldSchema::Enum {
                    enum_name: "RESULT".into(),
                    labels: vec!["MISS".into(), "HIT".into()],
                },
            ],
            field_names: Some(vec!["Roll".into(), "Result".into()]),
            probabilities: vec![(vec![20, 1], 1.0)],
        };
        assert_eq!(
            format_output_probabilities(enum_output)[0].0,
            "(Roll: 20, Result: HIT)"
        );
    }

    #[test]
    fn formats_structured_warnings_and_evaluation_traces() {
        let warning = Engine::new().run_with_diagnostics("D: d6\noutput D + D");
        let rendered = format_engine_diagnostics(&warning.diagnostics, &warning.sources);
        assert!(rendered.contains("evaluation.independent_pool_reuse"));
        assert!(rendered.contains("sampled independently"));

        let error = Engine::new().run_with_diagnostics(
            "function: pick I:n { result: [field I of [tuple 1 2]] } output [pick d3]",
        );
        let rendered = format_engine_diagnostics(&error.diagnostics, &error.sources);
        assert!(rendered.contains("while calling [pick …]"));
        assert!(rendered.contains("I = 3"));
    }
}
