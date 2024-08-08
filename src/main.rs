mod ast;
mod dice;
mod eval;
mod output;

use dice::Pool;
use lalrpop_util::lalrpop_mod;
use output::{export_anydice_format, print_diagnostic};

lalrpop_mod!(grammar);

fn main() {
    let parser = grammar::BodyParser::new();
    let mut rl = rustyline::DefaultEditor::new().unwrap();
    let mut evaluator = eval::Evaluator::new();
    while let Ok(line) = rl.readline("> ") {
        let statements = match parser.parse(&line) {
            Ok(expr) => expr,
            Err(err) => {
                eprintln!("Error: {}", err);
                continue;
            }
        };
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
                eval::RuntimeValue::Int(i) => Pool::from_list(1, vec![i]),
                eval::RuntimeValue::List(is) => Pool::from_list(1, is.to_vec()),
                eval::RuntimeValue::Pool(d) => (*d).clone(),
            };
            println!("{}", export_anydice_format(&name, &d));
        }
    }
}
