use eurydice_cli::{format_output_probabilities, print_engine_diagnostics};
use eurydice_engine::{Engine, PrintEvent};

fn main() {
    let mut rl = rustyline::DefaultEditor::new().unwrap();
    let mut engine = Engine::new();
    engine.set_print_callback(|event: PrintEvent| match event.name {
        Some(name) => println!("{}: {}", name, event.value),
        None => println!("{}", event.value),
    });
    let mut code = String::new();
    while let Ok(line) = rl.readline(if code.is_empty() { "> " } else { ". " }) {
        code.push_str(&line);
        let report = engine.run_source(&code);
        if report
            .diagnostics
            .first_error()
            .is_some_and(|error| error.code.is_incomplete())
        {
            code.push('\n');
            continue;
        }
        print_engine_diagnostics(&report.diagnostics);
        if report.diagnostics.first_error().is_some() {
            code.clear();
            continue;
        }
        let outputs = report.outputs;
        code.clear();

        for output in outputs {
            let (width, _) = crossterm::terminal::size().unwrap_or((80, 0));
            println!("{}:", output.name);
            let labeled_probabilities = format_output_probabilities(output.distribution);
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
