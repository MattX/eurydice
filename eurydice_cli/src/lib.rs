//! Terminal rendering helpers for engine outputs and diagnostics.
use std::collections::HashSet;

use eurydice_engine::{DiagnosticCode, DiagnosticSeverity, Diagnostics, Distribution, FieldSchema};
use miette::{
    Diagnostic as MietteDiagnostic, GraphicalReportHandler, LabeledSpan, NamedSource, Severity,
};

pub fn format_output_probabilities(distribution: Distribution) -> Vec<(String, f64)> {
    let is_tuple = distribution.fields.len() > 1;
    distribution
        .entries
        .into_iter()
        .map(|(outcome, probability)| {
            let fields = outcome
                .iter()
                .zip(&distribution.fields)
                .enumerate()
                .map(|(index, (value, schema))| {
                    let value = match schema {
                        FieldSchema::Int => value.to_string(),
                        FieldSchema::Categorical { labels, .. } => usize::try_from(*value)
                            .ok()
                            .and_then(|i| labels.get(i))
                            .cloned()
                            .unwrap_or_else(|| value.to_string()),
                        _ => value.to_string(),
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

/// Renders the engine's frontend-neutral diagnostics for terminal users.
pub fn format_engine_diagnostics(diagnostics: &Diagnostics) -> String {
    let sources = &diagnostics.sources;
    let handler = GraphicalReportHandler::new();
    let mut rendered = String::new();
    for diagnostic in &diagnostics.entries {
        let source_ids = diagnostic
            .labels()
            .map(|label| label.range.source)
            .collect::<HashSet<_>>();
        let mut first_source = true;
        for source in sources
            .iter()
            .filter(|source| source_ids.contains(&source.id))
        {
            let mut labels = Vec::new();
            if diagnostic.primary_label.range.source == source.id {
                let label = &diagnostic.primary_label;
                let span = (
                    label.range.range.start,
                    label.range.range.end - label.range.range.start,
                );
                labels.push(LabeledSpan::new_primary_with_span(
                    label.message.clone(),
                    span,
                ));
            }
            labels.extend(
                diagnostic
                    .secondary_labels
                    .iter()
                    .filter(|label| label.range.source == source.id)
                    .map(|label| {
                        let span = (
                            label.range.range.start,
                            label.range.range.end - label.range.range.start,
                        );
                        LabeledSpan::new_with_span(label.message.clone(), span)
                    }),
            );
            let adapter = MietteDiagnosticAdapter {
                summary: diagnostic.summary.clone(),
                code: diagnostic.code,
                severity: match diagnostic.code.severity() {
                    DiagnosticSeverity::Error => Severity::Error,
                    DiagnosticSeverity::Warning => Severity::Warning,
                    _ => Severity::Advice,
                },
                help: first_source.then(|| diagnostic.help.clone()).flatten(),
                source_code: NamedSource::new(source.name.clone(), source.text.clone()),
                labels,
            };
            let _ = handler.render_report(&mut rendered, &adapter);
            first_source = false;
        }

        if let Some(fix) = &diagnostic.fix {
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
                .map(|binding| format!("{} = {}", binding.name, binding.value))
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

pub fn print_engine_diagnostics(diagnostics: &Diagnostics) {
    eprint!("{}", format_engine_diagnostics(diagnostics));
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

struct MietteDiagnosticAdapter {
    summary: String,
    code: DiagnosticCode,
    severity: Severity,
    help: Option<String>,
    source_code: NamedSource<String>,
    labels: Vec<LabeledSpan>,
}

impl std::fmt::Display for MietteDiagnosticAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.summary)
    }
}

impl std::fmt::Debug for MietteDiagnosticAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MietteDiagnosticAdapter")
            .field("code", &self.code)
            .field("summary", &self.summary)
            .finish()
    }
}

impl std::error::Error for MietteDiagnosticAdapter {}

impl MietteDiagnostic for MietteDiagnosticAdapter {
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

#[cfg(test)]
mod tests {
    use super::{format_engine_diagnostics, format_output_probabilities};
    use eurydice_engine::Engine;
    use eurydice_engine::{Distribution, FieldSchema};

    #[test]
    fn formats_labeled_and_unlabeled_tuple_outcomes() {
        let output = |field_names| {
            Distribution::new(
                vec![FieldSchema::Int, FieldSchema::Int],
                field_names,
                vec![(vec![1, 2], 1.0)],
            )
        };

        assert_eq!(format_output_probabilities(output(None))[0].0, "(1, 2)");
        assert_eq!(
            format_output_probabilities(output(Some(vec!["A".into(), "B".into()])))[0].0,
            "(A: 1, B: 2)"
        );

        let enum_output = Distribution::new(
            vec![
                FieldSchema::Int,
                FieldSchema::Categorical {
                    labels: vec!["MISS".into(), "HIT".into()],
                },
            ],
            Some(vec!["Roll".into(), "Result".into()]),
            vec![(vec![20, 1], 1.0)],
        );
        assert_eq!(
            format_output_probabilities(enum_output)[0].0,
            "(Roll: 20, Result: HIT)"
        );
    }

    #[test]
    fn formats_structured_warnings_and_evaluation_traces() {
        let warning = Engine::new().run_source("D: d6\noutput D + D");
        let rendered = format_engine_diagnostics(&warning.diagnostics);
        assert!(rendered.contains("evaluation.independent_pool_reuse"));
        assert!(rendered.contains("sampled independently"));

        let error = Engine::new()
            .run_source("function: pick I:n { result: [field I of [tuple 1 2]] } output [pick d3]");
        let rendered = format_engine_diagnostics(&error.diagnostics);
        assert!(rendered.contains("while calling [pick …]"));
        assert!(rendered.contains("I = 3"));
    }
}
