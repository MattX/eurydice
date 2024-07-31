mod probability;
mod output;
mod parse;
mod lang;

use std::io::{self, BufRead as _, Read};

use output::print_distribution;
use probability::{Distribution, Outcome};

fn main() {
    // println!("Hello, world!");
    // let d6 = Distribution::uniform(1, 6);
    // let sum_of_three_d6 = Distribution::sum(vec![d6.clone(), d6.clone(), d6.clone()]).unwrap();
    // println!("Sum of three d6:");
    // print_distribution(&sum_of_three_d6);
    // let max_of_three_d6 = Distribution::max(vec![d6.clone(), d6.clone(), d6.clone()]).unwrap();
    // println!("Max of three d6:");
    // print_distribution(&max_of_three_d6);
    // let two_d6 = Distribution::cross_product(vec![d6.clone(), d6.clone()]);
    // // println!("Two d6:\n{}", two_d6.table());
    // let two_d6_sorted = two_d6.map(|outcome| {
    //     let mut sorted = outcome.0.clone();
    //     sorted.sort();
    //     Outcome(sorted)
    // });
    // println!("Two d6 sorted:");
    // print_distribution(&two_d6_sorted);

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let expr = match parse::parse(&line) {
            Ok(expr) => expr,
            Err(err) => {
                eprintln!("Error: {}", err);
                continue;
            }
        };
        println!("{}", parse::print_expression(&expr));
        let result = match lang::evaluate(&expr) {
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
