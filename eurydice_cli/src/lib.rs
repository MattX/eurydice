//! This contains graphical utilities used both by the CLI and the integ tests.
use eurydice_engine::output::{OutputValue, TupleFieldSchema};
use miette::{Diagnostic, GraphicalReportHandler};

pub fn format_output_probabilities(output: OutputValue) -> Vec<(String, f64)> {
    match output {
        OutputValue::Distribution(distribution) => distribution
            .probabilities
            .into_iter()
            .map(|(outcome, probability)| {
                let label = distribution
                    .labels
                    .as_ref()
                    .and_then(|labels| usize::try_from(outcome).ok().and_then(|i| labels.get(i)))
                    .cloned()
                    .unwrap_or_else(|| outcome.to_string());
                (label, probability)
            })
            .collect(),
        OutputValue::TupleDistribution(distribution) => distribution
            .probabilities
            .into_iter()
            .map(|(outcome, probability)| {
                let fields = outcome
                    .iter()
                    .zip(&distribution.fields)
                    .enumerate()
                    .map(|(index, (value, schema))| {
                        let value = match schema {
                            TupleFieldSchema::Int => value.to_string(),
                            TupleFieldSchema::Enum { labels, .. } => usize::try_from(*value)
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
                    .collect::<Vec<_>>()
                    .join(", ");
                (format!("({fields})"), probability)
            })
            .collect(),
    }
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
    use super::format_output_probabilities;
    use eurydice_engine::output::{OutputValue, TupleDistribution, TupleFieldSchema};

    #[test]
    fn formats_labeled_and_unlabeled_tuple_outcomes() {
        let output = |field_names| {
            OutputValue::TupleDistribution(TupleDistribution {
                fields: vec![TupleFieldSchema::Int, TupleFieldSchema::Int],
                field_names,
                probabilities: vec![(vec![1, 2], 1.0)],
            })
        };

        assert_eq!(format_output_probabilities(output(None))[0].0, "(1, 2)");
        assert_eq!(
            format_output_probabilities(output(Some(vec!["A".into(), "B".into()])))[0].0,
            "(A: 1, B: 2)"
        );

        let enum_output = OutputValue::TupleDistribution(TupleDistribution {
            fields: vec![
                TupleFieldSchema::Int,
                TupleFieldSchema::Enum {
                    enum_name: "RESULT".into(),
                    labels: vec!["MISS".into(), "HIT".into()],
                },
            ],
            field_names: Some(vec!["Roll".into(), "Result".into()]),
            probabilities: vec![(vec![20, 1], 1.0)],
        });
        assert_eq!(
            format_output_probabilities(enum_output)[0].0,
            "(Roll: 20, Result: HIT)"
        );
    }
}
