use approx::relative_ne;
use csv::ReaderBuilder;
use everydice::{
    dice::Pool,
    eval, grammar,
    output::{
        export_anydice_format, mean, min_and_max, print_diagnostic, stddev, to_probabilities,
    },
};
use miette::{Diagnostic, SourceSpan};
use pretty_assertions::StrComparison;
use std::{collections::HashSet, fs, path::Path};
use thiserror::Error;

#[test]
fn test_anydice_programs() {
    let test_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join("anydice");
    let mut paths_with_errors = HashSet::new();
    let mut paths = HashSet::new();
    for entry in fs::read_dir(test_dir).expect("Failed to read test directory") {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();
        let path_string = path.to_string_lossy().to_string();
        if path.file_name().unwrap().to_str().unwrap().starts_with('_') {
            continue;
        }

        if path.is_file() {
            paths.insert(path_string.clone());
            let content = fs::read_to_string(&path).expect("Failed to read file");
            let parts: Vec<&str> = content.split("\\\\\\ Result \\\\\\").collect();

            if parts.len() != 2 {
                paths_with_errors.insert(path_string);
                println!(
                    "Error: {} sections (2 expected) found in {}",
                    parts.len(),
                    path.display()
                );
                continue;
            }

            let program = parts[0].trim();
            let expected_results_strings = parts[1]
                .trim_start()
                .trim_end()
                .split("\n\n")
                .collect::<Vec<_>>();
            let expected_results = expected_results_strings
                .iter()
                .map(|part| parse_results(part))
                .collect::<Result<Vec<_>, _>>();
            let expected_results = match expected_results {
                Ok(results) => results,
                Err(err) => {
                    paths_with_errors.insert(path_string);
                    println!("CSV error in file {}: {}", path.display(), err);
                    continue;
                }
            };

            let mut evaluator = eval::Evaluator::new();
            let parser = grammar::BodyParser::new();

            let statements = match parser.parse(program) {
                Ok(expr) => expr,
                Err(err) => {
                    paths_with_errors.insert(path_string);
                    println!("Parsing error in file {}: {}", path.display(), err);
                    print_diagnostic(convert_lalrpop_error(&err), program);
                    continue;
                }
            };

            for statement in statements {
                match evaluator.execute(&statement) {
                    Ok(()) => {}
                    Err(e) => {
                        paths_with_errors.insert(path_string.clone());
                        println!("Evaluation error in file {}: {}", path.display(), e);
                        print_diagnostic(e, program);
                        continue;
                    }
                }
            }

            let outputs = evaluator.take_outputs();
            if outputs.len() != expected_results.len() {
                paths_with_errors.insert(path_string.clone());
                println!(
                    "Mismatch in file {}: expected {} outputs, got {}",
                    path.display(),
                    expected_results.len(),
                    evaluator.take_outputs().len()
                );
            }

            for (((value, name), expected), expected_str) in outputs
                .into_iter()
                .zip(expected_results.iter())
                .zip(expected_results_strings.iter())
            {
                let d = match value {
                    eval::RuntimeValue::Int(i) => Pool::from_list(1, vec![i]),
                    eval::RuntimeValue::List(is) => Pool::from_list(1, is.to_vec()),
                    eval::RuntimeValue::Pool(d) => (*d).clone(),
                };
                let actual_result = create_expected_result(&name, &d);
                if !compare_expected_results(&actual_result, expected) {
                    paths_with_errors.insert(path_string.clone());
                    println!("Mismatch in file {}:", path.display());
                    println!(
                        "{}",
                        StrComparison::new(&export_anydice_format(&name, &d), expected_str)
                    );
                }
            }
        }
    }
    println!("Paths without errors:");
    let mut sorted_ok_paths = paths.difference(&paths_with_errors).collect::<Vec<_>>();
    sorted_ok_paths.sort();
    for path in &sorted_ok_paths {
        println!("✅ {}", path);
    }
    println!("Paths with errors:");
    let mut sorted_paths_with_errors = paths_with_errors.iter().collect::<Vec<_>>();
    sorted_paths_with_errors.sort();
    for path in &sorted_paths_with_errors {
        println!("❌ {}", path);
    }
    assert_eq!(paths_with_errors.len(), 0, "Some test files had errors");
}

#[derive(Debug)]
struct ExpectedResult {
    name: String,
    mean: f64,
    stddev: f64,
    min: i32,
    max: i32,
    outcomes: Vec<(i32, f64)>,
}

fn parse_results(contents: &str) -> Result<ExpectedResult, Box<dyn std::error::Error>> {
    let mut lines = contents.lines();

    // Parse the first line
    let first_line = lines.next().ok_or("Empty file")?;
    let mut first_reader = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(first_line.as_bytes());
    let first_record = first_reader.records().next().ok_or("No data")??;

    let name = first_record
        .get(0)
        .ok_or("Missing name")?
        .trim_matches('"')
        .to_string();
    let mean = first_record.get(1).ok_or("Missing mean")?.parse()?;
    let stddev = first_record.get(2).ok_or("Missing stddev")?.parse()?;
    let min = first_record.get(3).ok_or("Missing min")?.parse()?;
    let max = first_record.get(4).ok_or("Missing max")?.parse()?;

    // Skip the "#,%" line
    let separator_line = lines.next().ok_or("Missing #,% line")?;
    if separator_line.trim() != "#,%" {
        return Err(Box::new(CsvError {
            desc: "missing '#,%' line".to_owned(),
        }));
    };

    // Parse the outcomes
    let mut outcomes = Vec::new();
    let rest = lines.collect::<Vec<&str>>().join("\n");
    let mut outcome_reader = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(rest.as_bytes());

    for result in outcome_reader.records() {
        let record = result?;
        let value = record.get(0).ok_or("Missing outcome value")?.parse()?;
        let probability = record
            .get(1)
            .ok_or("Missing outcome probability")?
            .parse()?;
        outcomes.push((value, probability));
    }

    Ok(ExpectedResult {
        name,
        mean,
        stddev,
        min,
        max,
        outcomes,
    })
}

fn create_expected_result(name: &str, pool: &Pool) -> ExpectedResult {
    let probabilities = to_probabilities(pool.sum().ordered_outcomes());
    let mean = mean(&probabilities);
    let stddev = stddev(&probabilities, mean);
    let (min, max) = min_and_max(&probabilities);

    let mut outcomes = Vec::new();
    for (outcome, prob) in probabilities {
        outcomes.push((outcome, prob * 100.0));
    }
    ExpectedResult {
        name: name.to_string(),
        mean,
        stddev,
        min,
        max,
        outcomes,
    }
}

fn compare_expected_results(a: &ExpectedResult, b: &ExpectedResult) -> bool {
    const EPSILON: f64 = 1e-6;

    // Compare name
    if a.name != b.name {
        return false;
    }

    // Compare mean
    if relative_ne!(a.mean, b.mean, epsilon = EPSILON, max_relative = EPSILON) {
        return false;
    }

    // Compare stddev
    if relative_ne!(
        a.stddev,
        b.stddev,
        epsilon = EPSILON,
        max_relative = EPSILON
    ) {
        return false;
    }

    // Compare min and max
    if a.min != b.min {
        return false;
    }
    if a.max != b.max {
        return false;
    }

    // Compare outcomes
    if a.outcomes.len() != b.outcomes.len() {
        return false;
    }

    for ((a_value, a_prob), (b_value, b_prob)) in a.outcomes.iter().zip(b.outcomes.iter()) {
        if a_value != b_value {
            return false;
        }
        if relative_ne!(a_prob, b_prob, epsilon = EPSILON, max_relative = EPSILON) {
            return false;
        }
    }

    true
}

#[derive(Error, Diagnostic, Debug)]
#[error("csv error")]
struct CsvError {
    desc: String,
}

#[derive(Error, Diagnostic, Debug)]
#[error("parse error")]
struct ParseError {
    #[label("parse error: {desc}")]
    span: SourceSpan,
    desc: String,
}

impl ParseError {
    fn new(span: SourceSpan, desc: String) -> Self {
        Self { span, desc }
    }
}

fn convert_lalrpop_error<T: std::fmt::Display, E: std::fmt::Display>(
    err: &lalrpop_util::ParseError<usize, T, E>,
) -> ParseError {
    match err {
        lalrpop_util::ParseError::InvalidToken { location } => {
            ParseError::new(SourceSpan::new((*location).into(), 0), format!("{}", err))
        }
        lalrpop_util::ParseError::UnrecognizedEof { location, .. } => {
            ParseError::new(SourceSpan::new((*location).into(), 0), format!("{}", err))
        }
        lalrpop_util::ParseError::UnrecognizedToken { token, .. } => ParseError::new(
            SourceSpan::new(token.0.into(), token.2 - token.0),
            format!("{}", err),
        ),
        lalrpop_util::ParseError::ExtraToken { token } => ParseError::new(
            SourceSpan::new(token.0.into(), token.2 - token.0),
            format!("{}", err),
        ),
        lalrpop_util::ParseError::User { error } => {
            ParseError::new(SourceSpan::new(0.into(), 0), format!("{}", error))
        }
    }
}
