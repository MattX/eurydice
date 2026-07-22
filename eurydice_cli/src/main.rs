use eurydice_cli::print_diagnostic;
use eurydice_engine::output::{OutputValue, TupleFieldSchema};
use lalrpop_util::ParseError;

fn main() {
    let parser = eurydice_engine::grammar::BodyParser::new();
    let mut rl = rustyline::DefaultEditor::new().unwrap();
    let mut evaluator = eurydice_engine::eval::Evaluator::new();
    evaluator.set_print_callback(Box::new(|value, name| {
        println!("{}: {}", name, value);
    }));
    let mut code = String::new();
    while let Ok(line) = rl.readline(if code.is_empty() { "> " } else { ". " }) {
        code.push_str(&line);
        let statements = match parser.parse(&code) {
            Ok(expr) => expr,
            Err(ParseError::UnrecognizedEof { .. }) => {
                continue;
            }
            Err(err) => {
                eprintln!("Error: {}", err);
                code.clear();
                continue;
            }
        };
        code.clear();

        for statement in statements {
            match evaluator.execute(&statement) {
                Ok(()) => {}
                Err(e) => {
                    print_diagnostic(e, &line);
                    continue;
                }
            }
        }
        for (value, name) in evaluator.take_outputs() {
            let (width, _) = crossterm::terminal::size().unwrap_or((80, 0));
            println!("{}:", name);
            let labeled_probabilities: Vec<(String, f64)> = match OutputValue::from(value) {
                OutputValue::Distribution(distribution) => distribution
                    .probabilities
                    .into_iter()
                    .map(|(outcome, probability)| {
                        let label = distribution
                            .labels
                            .as_ref()
                            .and_then(|labels| {
                                usize::try_from(outcome).ok().and_then(|i| labels.get(i))
                            })
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
                            .map(|(value, schema)| match schema {
                                TupleFieldSchema::Int => value.to_string(),
                                TupleFieldSchema::Enum { labels, .. } => usize::try_from(*value)
                                    .ok()
                                    .and_then(|i| labels.get(i))
                                    .cloned()
                                    .unwrap_or_else(|| value.to_string()),
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        (format!("({fields})"), probability)
                    })
                    .collect(),
            };
            display_distribution(&labeled_probabilities, width);
        }
    }
}

fn display_distribution(distribution: &[(String, f64)], max_width: u16) {
    if distribution.is_empty() {
        println!("Distribution is empty");
        return;
    }

    let max_prob = *distribution
        .iter()
        .map(|(_outcome, prob)| prob)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let (label_width, max_bar_width) = distribution_layout(distribution, max_width);

    for (outcome, prob) in distribution {
        let bar_width = ((prob / max_prob) * max_bar_width as f64) as usize;
        let bar = "━".repeat(bar_width);
        println!("{:>label_width$} {:8.3}% |{}", outcome, prob * 100.0, bar);
    }
}

fn distribution_layout(distribution: &[(String, f64)], max_width: u16) -> (usize, usize) {
    const PROBABILITY_AND_SEPARATOR_WIDTH: usize = 12;

    let label_width = distribution
        .iter()
        .map(|(outcome, _prob)| outcome.chars().count())
        .max()
        .unwrap_or_default();
    let max_bar_width =
        usize::from(max_width).saturating_sub(label_width + PROBABILITY_AND_SEPARATOR_WIDTH);

    (label_width, max_bar_width)
}

#[cfg(test)]
mod tests {
    use super::distribution_layout;

    #[test]
    fn label_column_uses_widest_label_width() {
        let distribution = [("1".to_owned(), 0.5), ("three".to_owned(), 0.5)];

        assert_eq!(distribution_layout(&distribution, 80), (5, 63));
    }

    #[test]
    fn bar_width_accounts_for_wide_labels() {
        let distribution = [("(success, failure)".to_owned(), 1.0)];

        assert_eq!(distribution_layout(&distribution, 80), (18, 50));
    }

    #[test]
    fn narrow_terminal_has_zero_width_bar() {
        let distribution = [("(success, failure)".to_owned(), 1.0)];

        assert_eq!(distribution_layout(&distribution, 20), (18, 0));
    }
}
