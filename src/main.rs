mod probability;
mod output;

use output::print_distribution;
use probability::{Distribution, Outcome};

fn main() {
    println!("Hello, world!");
    let d6 = Distribution::uniform(1, 6);
    let sum_of_three_d6 = Distribution::sum(vec![d6.clone(), d6.clone(), d6.clone()]).unwrap();
    println!("Sum of three d6:");
    print_distribution(&sum_of_three_d6);
    let max_of_three_d6 = Distribution::max(vec![d6.clone(), d6.clone(), d6.clone()]).unwrap();
    println!("Max of three d6:");
    print_distribution(&max_of_three_d6);
    let two_d6 = Distribution::cross_product(vec![d6.clone(), d6.clone()]);
    // println!("Two d6:\n{}", two_d6.table());
    let two_d6_sorted = two_d6.map(|outcome| {
        let mut sorted = outcome.0.clone();
        sorted.sort();
        Outcome(sorted)
    });
    println!("Two d6 sorted:");
    print_distribution(&two_d6_sorted);
}
