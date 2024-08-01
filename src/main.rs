mod ast;
mod lang;
mod output;
// mod parse;
mod probability;
mod typecheck;

use std::io::{self, BufRead as _, BufWriter, Read};

use lalrpop_util::lalrpop_mod;
use miette::{GraphicalReportHandler, IntoDiagnostic};
use output::print_distribution;
use probability::{Distribution, Outcome};

lalrpop_mod!(
    #[allow(clippy::ptr_arg)]
    #[rustfmt::skip]
    grammar
);

fn main() {
    let stdin = io::stdin();
    let parser = grammar::ExprParser::new();
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let expr = match parser.parse(&line) {
            Ok(expr) => expr,
            Err(err) => {
                eprintln!("Error: {}", err);
                continue;
            }
        };
        println!("{}", ast::print_expression(&expr));
        let result = match lang::Evaluator::new(&line).evaluate(&expr) {
            Ok(distributions) => distributions,
            Err(e) => {
                let handler = GraphicalReportHandler::new();
                let mut string = String::new();
                let _ = handler.render_report(&mut string, &e);
                eprintln!("{}", string);
                continue;
            }
        };
        for distribution in result {
            print_distribution(&distribution);
        }
    }
}
