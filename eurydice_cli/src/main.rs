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

    for (outcome, prob) in distribution {
        let bar_width = ((prob / max_prob) * (max_width - 20) as f64) as u16;
        let bar = "━".repeat(bar_width as usize);
        println!("{:>12} {:8.3}% |{}", outcome, prob * 100.0, bar);
    }
}
