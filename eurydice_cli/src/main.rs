use eurydice_engine::dice::Pool;
use eurydice_engine::output::{export_anydice_format, print_diagnostic};
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
                eurydice_engine::eval::RuntimeValue::Pool(d) => (*d).clone(),
            };
            println!("{}", export_anydice_format(&name, &d));
        }
    }
}
