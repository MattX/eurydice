use eurydice_cli::print_diagnostic;
use eurydice_engine::dice::Pool;
use eurydice_engine::eval::{RuntimeValue, ScalarType, ScalarValue};
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
            let outcome_type = match &value {
                RuntimeValue::Scalar(value) => value.scalar_type(),
                RuntimeValue::List(_, ty) | RuntimeValue::Pool(_, ty) => ty.clone(),
            };
            let labels = match &outcome_type {
                ScalarType::Enum(ty) => Some(ty.members.clone()),
                ScalarType::Int | ScalarType::Tuple(_) => None,
            };
            let d = match value {
                RuntimeValue::Scalar(value) => Pool::from_list(1, vec![value]),
                RuntimeValue::List(values, _) => Pool::from_list(1, values.to_vec()),
                RuntimeValue::Pool(d, _) if outcome_type == ScalarType::Int => d
                    .as_ref()
                    .clone()
                    .map_outcomes(|value| value.as_int().expect("numeric pool"))
                    .sum()
                    .map_outcomes(ScalarValue::Int),
                RuntimeValue::Pool(d, _) => d.as_ref().clone(),
            };
            let (width, _) = crossterm::terminal::size().unwrap_or((80, 0));
            let dist = eurydice_engine::output::to_probabilities_generic(d.ordered_outcomes());
            println!("{}:", name);
            display_distribution(&dist, labels.as_deref(), width);
        }
    }
}

fn display_distribution(
    distribution: &[(ScalarValue, f64)],
    labels: Option<&[String]>,
    max_width: u16,
) {
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
        let label = labels
            .and_then(|labels| {
                outcome
                    .as_int()
                    .and_then(|outcome| usize::try_from(outcome).ok())
                    .and_then(|i| labels.get(i))
            })
            .cloned()
            .unwrap_or_else(|| outcome.to_string());
        println!("{:>12} {:8.3}% |{}", label, prob * 100.0, bar);
    }
}
