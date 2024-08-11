use eurydice_cli::print_diagnostic;
use eurydice_engine::dice::Pool;
use lalrpop_util::ParseError;

fn main() {
    let parser = eurydice_engine::grammar::BodyParser::new();
    let mut rl = rustyline::DefaultEditor::new().unwrap();
    let mut evaluator = eurydice_engine::eval::Evaluator::new();
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
            let d = match value {
                eurydice_engine::eval::RuntimeValue::Int(i) => Pool::from_list(1, vec![i]),
                eurydice_engine::eval::RuntimeValue::List(is) => Pool::from_list(1, is.to_vec()),
                eurydice_engine::eval::RuntimeValue::Pool(d) => (*d).clone().sum(),
            };
            let (width, _) = crossterm::terminal::size().unwrap_or((80, 0));
            let dist = eurydice_engine::output::to_probabilities(d.ordered_outcomes());
            println!("{}:", name);
            display_distribution(&dist, width);
        }
    }
}

fn display_distribution(distribution: &[(i32, f64)], max_width: u16) {
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
        println!("{:4} {:8.3}% |{}", outcome, prob * 100.0, bar);
    }
}
