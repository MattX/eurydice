mod ast;
mod lang;
mod output;
// mod parse;
mod probability;

use std::io::{self, BufRead as _, Read};

use lalrpop_util::lalrpop_mod;
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
        // println!("{}", parse::print_expression(&expr));
        let result = match lang::Evaluator::new(&line).evaluate(&expr) {
            Ok(distributions) => distributions,
            Err(err) => {
                eprintln!("Error: {}", err);
                continue;
            }
        };
        for distribution in result {
            print_distribution(&distribution);
        }
    }
}
