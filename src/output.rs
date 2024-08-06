use malachite::{num::basic::traits::Zero, Natural, Rational};
use miette::{Diagnostic, GraphicalReportHandler};
use std::fmt::Write;

use crate::dice::Pool;

pub fn print_distribution(pool: &Pool) {
    let mut outcome = Vec::new();
    let mut counts = Vec::new();
    let mut probs = Vec::new();

    let mut total = Natural::ZERO;
    for (out, count) in pool.ordered_outcomes().iter() {
        outcome.push(out.to_string());
        total += count;
        counts.push(count);
    }
    for count in counts {
        probs.push(format!(
            "{:.02}",
            Rational::from_naturals(count.clone(), total.clone())
                .approx_log()
                .exp()
        ));
    }
    println!("Pool of {} dice:", pool.get_n());
    Table {
        first_row_is_header: true,
        first_column_is_header: false,
        data: vec![outcome, probs],
    }
    .print();
}

pub fn export_anydice_format(name: &str, pool: &Pool) -> String {
    let probabilities = to_probabilities(pool.sum().ordered_outcomes());
    let mean = mean(&probabilities);
    let stddev = stddev(&probabilities, mean);
    let (min, max) = min_and_max(&probabilities);

    let mut string = String::new();
    writeln!(string, "\"{}\",{},{},{},{}", name, mean, stddev, min, max).unwrap();
    writeln!(string, "#,%").unwrap();
    for (outcome, prob) in probabilities {
        writeln!(string, "{},{}", outcome, prob * 100.0).unwrap();
    }
    string
}

pub fn print_diagnostic<T>(diagnostic: T, source_code: &str)
where
    T: Diagnostic,
{
    let handler = GraphicalReportHandler::new();
    let mut string = String::new();
    let wrapped = DiagnosticSourceAdapter {
        source: diagnostic,
        source_code,
    };
    let _ = handler.render_report(&mut string, &wrapped);
    eprintln!("{}", string);
}

fn to_probabilities(ordered_outcomes: &[(i32, Natural)]) -> Vec<(i32, f64)> {
    let total: Natural = ordered_outcomes.iter().map(|(_, count)| count).sum();
    ordered_outcomes
        .iter()
        .map(|(outcome, count)| {
            (
                *outcome,
                Rational::from_naturals(count.clone(), total.clone())
                    .approx_log()
                    .exp(),
            )
        })
        .collect()
}

fn mean(probabilities: &[(i32, f64)]) -> f64 {
    probabilities
        .iter()
        .map(|(outcome, prob)| *outcome as f64 * *prob)
        .sum()
}

fn stddev(probabilities: &[(i32, f64)], mean: f64) -> f64 {
    let variance: f64 = probabilities
        .iter()
        .map(|(outcome, prob)| (*outcome as f64 - mean).powi(2) * *prob)
        .sum();
    variance.sqrt()
}

fn min_and_max(probabilities: &[(i32, f64)]) -> (i32, i32) {
    let min = probabilities.iter().map(|(outcome, _)| *outcome).min();
    let max = probabilities.iter().map(|(outcome, _)| *outcome).max();
    (min.unwrap(), max.unwrap())
}

#[derive(Clone)]
struct Table {
    pub first_row_is_header: bool,
    pub first_column_is_header: bool,
    pub data: Vec<Vec<String>>, // Row-major order
}

impl Table {
    pub fn print(&self) {
        if self.data.is_empty() {
            return;
        }
        let column_count = self.data[0].len();
        if self.data.iter().any(|row| row.len() != column_count) {
            panic!("All rows must have the same number of columns");
        }
        if column_count == 0 {
            return;
        }

        // Determine each column's width
        let column_widths: Vec<usize> = (0..self.data[0].len())
            .map(|column| self.data.iter().map(|row| row[column].len()).max().unwrap())
            .collect();

        let terminal_width = crossterm::terminal::size().map(|(w, _)| w).unwrap_or(80);
        let mut next_column = if self.first_column_is_header { 1 } else { 0 };
        while next_column < column_count {
            let columns_to_print = columns_to_print(
                self.first_column_is_header,
                terminal_width,
                &column_widths,
                next_column,
            );
            let this_chunk_column_widths =
                &column_widths[next_column..next_column + columns_to_print];
            print_row_separator(this_chunk_column_widths, SeparatorKind::Top);
            if self.first_row_is_header {
                print_row(
                    &self.data[0][next_column..next_column + columns_to_print],
                    this_chunk_column_widths,
                );
                print_row_separator(this_chunk_column_widths, SeparatorKind::Middle);
            }
            for row in self
                .data
                .iter()
                .skip(if self.first_row_is_header { 1 } else { 0 })
            {
                print_row(
                    &row[next_column..next_column + columns_to_print],
                    this_chunk_column_widths,
                );
            }
            print_row_separator(this_chunk_column_widths, SeparatorKind::Bottom);
            next_column += columns_to_print;
        }
    }
}

fn columns_to_print(
    first_column_is_header: bool,
    terminal_width: u16,
    column_widths: &[usize],
    next_column: usize,
) -> usize {
    let number_of_columns = column_widths.len();
    let mut chars_taken_up = 4; // 2 chars on each side of the row
    if first_column_is_header {
        chars_taken_up += column_widths[0] + 3;
    }
    chars_taken_up += column_widths[next_column];
    let mut to_print = 1; // Minimum of 1 column (+ column header) in order to make progress
    while next_column + to_print < number_of_columns
        && chars_taken_up + 3 + column_widths[next_column + to_print] <= terminal_width as usize
    {
        chars_taken_up += 3 + column_widths[next_column + to_print];
        to_print += 1;
    }
    to_print
}

#[derive(Debug, Copy, Clone)]
enum SeparatorKind {
    Top,
    Middle,
    Bottom,
}

fn print_row_separator(column_widths: &[usize], kind: SeparatorKind) {
    let (left_separator, center_separator, right_separator) = match kind {
        SeparatorKind::Top => ('╭', '┬', '╮'),
        SeparatorKind::Middle => ('├', '┼', '┤'),
        SeparatorKind::Bottom => ('╰', '┴', '╯'),
    };
    print!("{}", left_separator);
    for column in 0..column_widths.len() {
        print!("{}", "─".repeat(column_widths[column] + 2));
        print!(
            "{}",
            if column == column_widths.len() - 1 {
                right_separator
            } else {
                center_separator
            }
        );
    }
    println!();
}

fn print_row(row: &[String], column_widths: &[usize]) {
    print!("│");
    for (column, width) in row.iter().zip(column_widths) {
        print!(" {}{} ", " ".repeat(width - column.len()), column);
        print!("│");
    }
    println!();
}

struct DiagnosticSourceAdapter<T, U: miette::SourceCode>
where
    T: std::error::Error,
    T: Diagnostic,
{
    source: T,
    source_code: U,
}

impl<T, U> std::fmt::Display for DiagnosticSourceAdapter<T, U>
where
    T: std::error::Error,
    T: Diagnostic,
    U: miette::SourceCode,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.source)
    }
}

impl<T, U> std::fmt::Debug for DiagnosticSourceAdapter<T, U>
where
    T: std::error::Error,
    T: Diagnostic,
    U: miette::SourceCode,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.source)
    }
}

impl<T, U> std::error::Error for DiagnosticSourceAdapter<T, U>
where
    T: std::error::Error,
    T: Diagnostic,
    U: miette::SourceCode,
{
}

impl<T, U> Diagnostic for DiagnosticSourceAdapter<T, U>
where
    T: std::error::Error,
    T: Diagnostic,
    U: miette::SourceCode,
{
    fn source_code(&self) -> Option<&dyn miette::SourceCode> {
        Some(&self.source_code)
    }

    fn code<'a>(&'a self) -> Option<Box<dyn std::fmt::Display + 'a>> {
        self.source.code()
    }

    fn severity(&self) -> Option<miette::Severity> {
        self.source.severity()
    }

    fn help<'a>(&'a self) -> Option<Box<dyn std::fmt::Display + 'a>> {
        self.source.help()
    }

    fn url<'a>(&'a self) -> Option<Box<dyn std::fmt::Display + 'a>> {
        self.source.url()
    }

    fn labels(&self) -> Option<Box<dyn Iterator<Item = miette::LabeledSpan> + '_>> {
        self.source.labels()
    }

    fn related<'a>(&'a self) -> Option<Box<dyn Iterator<Item = &'a dyn Diagnostic> + 'a>> {
        self.source.related()
    }

    fn diagnostic_source(&self) -> Option<&dyn Diagnostic> {
        self.source.diagnostic_source()
    }
}
