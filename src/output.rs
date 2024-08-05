// use crate::probability::Distribution;

// pub fn print_distribution(distribution: &Distribution) {
//     let mut outcome = Vec::new();
//     let mut probs = Vec::new();

//     for (out, prob) in &distribution.probabilities {
//         outcome.push(
//             out.0
//                 .iter()
//                 .map(|&x| x.to_string())
//                 .collect::<Vec<String>>()
//                 .join(","),
//         );
//         probs.push(format!("{:.2}", prob));
//     }
//     Table {
//         first_row_is_header: true,
//         first_column_is_header: false,
//         data: vec![outcome, probs],
//     }
//     .print();
// }

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
