//! Full-program golden tests.
//!
//! AnyDice compatibility fixtures retain AnyDice's numeric summary line:
//! `"name",mean,stddev,min,max`. Eurydice-native fixtures use just `"name"`,
//! followed by one header per output field and a final `%` column. `#` is an
//! integer field, `Symbol` marks an enum field, and labeled tuple fields use
//! their labels. Outcome rows contain rendered field values and a percentage.

use approx::relative_ne;
use csv::{ReaderBuilder, WriterBuilder};
use eurydice_engine::{Distribution, Engine, FieldSchema};
use pretty_assertions::StrComparison;
use std::{collections::HashSet, fmt::Write, fs, path::Path};
use thiserror::Error;

/// Fixtures whose expected output was captured from AnyDice itself, and so
/// double as compatibility tests.
#[test]
fn test_anydice_programs() {
    run_fixture_directory("anydice");
}

/// Fixtures for Eurydice-only features, which have no AnyDice equivalent.
/// Their expected output is recorded from this implementation rather than
/// from AnyDice, so they guard against regressions rather than proving
/// compatibility.
#[test]
fn test_eurydice_programs() {
    run_fixture_directory("eurydice");
}

fn run_fixture_directory(directory: &str) {
    let test_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(directory);
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

            let report = Engine::new().run_source(program);
            if let Some(error) = report.error() {
                paths_with_errors.insert(path_string);
                println!(
                    "Error in file {}: [{}] {}",
                    path.display(),
                    error.code,
                    error.summary
                );
                continue;
            }

            let outputs = report.outputs;
            if outputs.len() != expected_results.len() {
                paths_with_errors.insert(path_string.clone());
                println!(
                    "Mismatch in file {}: expected {} outputs, got {}",
                    path.display(),
                    expected_results.len(),
                    outputs.len()
                );
            }

            for ((output, expected), expected_str) in outputs
                .into_iter()
                .zip(expected_results.iter())
                .zip(expected_results_strings.iter())
            {
                let mismatch = match expected {
                    ExpectedResult::AnyDice(expected) => {
                        let actual = match create_anydice_result(&output.name, &output.distribution)
                        {
                            Ok(actual) => actual,
                            Err(error) => {
                                paths_with_errors.insert(path_string.clone());
                                println!(
                                    "Unsupported AnyDice output in file {}: {}",
                                    path.display(),
                                    error
                                );
                                continue;
                            }
                        };
                        (!compare_anydice_results(&actual, expected))
                            .then(|| export_anydice_result(&actual))
                    }
                    ExpectedResult::Distribution(expected) => {
                        let actual = create_distribution_result(&output.name, output.distribution);
                        (!compare_distribution_results(&actual, expected))
                            .then(|| export_distribution_result(&actual))
                    }
                };

                if let Some(actual_string) = mismatch {
                    paths_with_errors.insert(path_string.clone());
                    println!("Mismatch in file {}:", path.display());
                    println!("{}", StrComparison::new(&actual_string, expected_str));
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
enum ExpectedResult {
    AnyDice(AnyDiceResult),
    Distribution(DistributionResult),
}

#[derive(Debug)]
struct AnyDiceResult {
    name: String,
    mean: f64,
    stddev: f64,
    min: i32,
    max: i32,
    outcomes: Vec<(i32, f64)>,
}

#[derive(Debug)]
struct DistributionResult {
    name: String,
    fields: Vec<String>,
    outcomes: Vec<(Vec<String>, f64)>,
}

fn mean(probabilities: &[(i32, f64)]) -> f64 {
    probabilities
        .iter()
        .map(|(outcome, probability)| f64::from(*outcome) * probability)
        .sum()
}

fn stddev(probabilities: &[(i32, f64)], mean: f64) -> f64 {
    probabilities
        .iter()
        .map(|(outcome, probability)| (f64::from(*outcome) - mean).powi(2) * probability)
        .sum::<f64>()
        .sqrt()
}

fn min_and_max(probabilities: &[(i32, f64)]) -> (i32, i32) {
    let mut outcomes = probabilities.iter().map(|(outcome, _)| *outcome);
    let Some(first) = outcomes.next() else {
        return (0, 0);
    };
    outcomes.fold((first, first), |(min, max), outcome| {
        (min.min(outcome), max.max(outcome))
    })
}

fn parse_results(contents: &str) -> Result<ExpectedResult, Box<dyn std::error::Error>> {
    let mut lines = contents.lines();

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

    match first_record.len() {
        5 => parse_anydice_result(name, &first_record, lines),
        1 => parse_distribution_result(name, lines),
        count => Err(Box::new(CsvError {
            desc: format!(
                "result heading must contain either a name or a name plus four statistics; found {count} fields"
            ),
        })),
    }
}

fn parse_anydice_result<'a>(
    name: String,
    first_record: &csv::StringRecord,
    mut lines: impl Iterator<Item = &'a str>,
) -> Result<ExpectedResult, Box<dyn std::error::Error>> {
    let mean = first_record.get(1).ok_or("Missing mean")?.parse()?;
    let stddev = first_record.get(2).ok_or("Missing stddev")?.parse()?;
    let min = first_record.get(3).ok_or("Missing min")?.parse()?;
    let max = first_record.get(4).ok_or("Missing max")?.parse()?;

    let separator_line = lines.next().ok_or("Missing #,% line")?;
    if separator_line.trim() != "#,%" {
        return Err(Box::new(CsvError {
            desc: "missing '#,%' line".to_owned(),
        }));
    };

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

    Ok(ExpectedResult::AnyDice(AnyDiceResult {
        name,
        mean,
        stddev,
        min,
        max,
        outcomes,
    }))
}

fn parse_distribution_result<'a>(
    name: String,
    lines: impl Iterator<Item = &'a str>,
) -> Result<ExpectedResult, Box<dyn std::error::Error>> {
    let rest = lines.collect::<Vec<&str>>().join("\n");
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(rest.as_bytes());
    let mut records = reader.records();
    let header = records.next().ok_or("Missing distribution header")??;
    if header.len() < 2 || header.get(header.len() - 1) != Some("%") {
        return Err(Box::new(CsvError {
            desc: "distribution header must end with '%'".to_owned(),
        }));
    }
    let fields = header
        .iter()
        .take(header.len() - 1)
        .map(str::to_owned)
        .collect::<Vec<_>>();

    let mut outcomes = Vec::new();
    for result in records {
        let record = result?;
        if record.len() != fields.len() + 1 {
            return Err(Box::new(CsvError {
                desc: format!(
                    "expected {} outcome fields plus a probability, found {} columns",
                    fields.len(),
                    record.len()
                ),
            }));
        }
        let values = record
            .iter()
            .take(fields.len())
            .map(str::to_owned)
            .collect();
        let probability = record
            .get(fields.len())
            .ok_or("Missing outcome probability")?
            .parse()?;
        outcomes.push((values, probability));
    }

    Ok(ExpectedResult::Distribution(DistributionResult {
        name,
        fields,
        outcomes,
    }))
}

/// Reads an AnyDice-comparable result out of the distribution a frontend would
/// receive.
///
/// This reads the [`Distribution`] the engine itself produced rather than
/// summing the pool. A harness that converts outputs its own way can only test
/// its own conversion: it once rendered `[explode d{}]` as no outcomes while the
/// real engine rendered `0` at 100%, and the fixture asserting the AnyDice
/// answer passed regardless.
///
/// An empty distribution needs no special case — it has no outcomes, and the
/// statistics of nothing are the zeroes AnyDice reports.
fn create_anydice_result(
    name: &str,
    distribution: &Distribution,
) -> Result<AnyDiceResult, &'static str> {
    match distribution.fields.as_slice() {
        [FieldSchema::Int] => {}
        [FieldSchema::Categorical { .. }] => {
            return Err("AnyDice fixtures must have numeric outputs");
        }
        _ => return Err("AnyDice fixtures must have a single output field"),
    }
    let probabilities = distribution
        .entries
        .iter()
        .map(|(values, probability)| (values[0], *probability))
        .collect::<Vec<_>>();
    let mean = mean(&probabilities);
    let stddev = stddev(&probabilities, mean);
    let (min, max) = min_and_max(&probabilities);

    Ok(AnyDiceResult {
        name: name.to_string(),
        mean,
        stddev,
        min,
        max,
        outcomes: probabilities
            .into_iter()
            .map(|(outcome, probability)| (outcome, probability * 100.0))
            .collect(),
    })
}

/// Renders a mismatched result in the fixture's own format, for the diff.
fn export_anydice_result(result: &AnyDiceResult) -> String {
    let mut string = String::new();
    writeln!(
        string,
        "\"{}\",{},{},{},{}",
        result.name, result.mean, result.stddev, result.min, result.max
    )
    .expect("write to string");
    writeln!(string, "#,%").expect("write to string");
    for (outcome, percent) in &result.outcomes {
        writeln!(string, "{outcome},{percent}").expect("write to string");
    }
    string
}

fn create_distribution_result(name: &str, distribution: Distribution) -> DistributionResult {
    let fields = distribution.field_names.unwrap_or_else(|| {
        distribution
            .fields
            .iter()
            .map(|field| match field {
                FieldSchema::Int => "#".to_owned(),
                FieldSchema::Categorical { .. } => "Symbol".to_owned(),
                _ => "Value".to_owned(),
            })
            .collect()
    });
    let outcomes = distribution
        .entries
        .into_iter()
        .map(|(values, probability)| {
            let values = values
                .into_iter()
                .zip(&distribution.fields)
                .map(|(value, field)| match field {
                    FieldSchema::Int => value.to_string(),
                    FieldSchema::Categorical { labels, .. } => usize::try_from(value)
                        .ok()
                        .and_then(|index| labels.get(index))
                        .cloned()
                        .unwrap_or_else(|| value.to_string()),
                    _ => value.to_string(),
                })
                .collect();
            (values, probability * 100.0)
        })
        .collect();
    DistributionResult {
        name: name.to_owned(),
        fields,
        outcomes,
    }
}

fn compare_anydice_results(a: &AnyDiceResult, b: &AnyDiceResult) -> bool {
    const EPSILON: f64 = 1e-6;

    if a.name != b.name {
        return false;
    }

    if relative_ne!(a.mean, b.mean, epsilon = EPSILON, max_relative = EPSILON) {
        return false;
    }

    if relative_ne!(
        a.stddev,
        b.stddev,
        epsilon = EPSILON,
        max_relative = EPSILON
    ) {
        return false;
    }

    if a.min != b.min {
        return false;
    }
    if a.max != b.max {
        return false;
    }

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

fn compare_distribution_results(a: &DistributionResult, b: &DistributionResult) -> bool {
    const EPSILON: f64 = 1e-6;

    a.name == b.name
        && a.fields == b.fields
        && a.outcomes.len() == b.outcomes.len()
        && a.outcomes.iter().zip(&b.outcomes).all(
            |((a_values, a_probability), (b_values, b_probability))| {
                a_values == b_values
                    && !relative_ne!(
                        a_probability,
                        b_probability,
                        epsilon = EPSILON,
                        max_relative = EPSILON
                    )
            },
        )
}

fn export_distribution_result(result: &DistributionResult) -> String {
    // The name row has one field and the header has one per output field, so the
    // writer must accept records of differing lengths.
    let mut writer = WriterBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_writer(Vec::new());
    writer.write_record([&result.name]).unwrap();
    let mut header = result.fields.clone();
    header.push("%".to_owned());
    writer.write_record(header).unwrap();
    for (values, probability) in &result.outcomes {
        let mut record = values.clone();
        record.push(probability.to_string());
        writer.write_record(record).unwrap();
    }
    String::from_utf8(writer.into_inner().unwrap()).unwrap()
}

#[derive(Error, Debug)]
#[error("csv error: {desc}")]
struct CsvError {
    desc: String,
}
