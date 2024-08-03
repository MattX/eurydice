mod ast;
mod dice;
mod eval;
mod output;
mod probability;
mod typecheck;
mod util;

use lalrpop_util::lalrpop_mod;
use miette::{Diagnostic, GraphicalReportHandler};
use output::print_distribution;

lalrpop_mod!(grammar);

fn main() {
    let parser = grammar::ExprParser::new();
    let mut rl = rustyline::DefaultEditor::new().unwrap();
    while let Ok(line) = rl.readline("> ") {
        let expr = match parser.parse(&line) {
            Ok(expr) => expr,
            Err(err) => {
                eprintln!("Error: {}", err);
                continue;
            }
        };
        println!("{}", ast::print_expression(&expr));
        match typecheck::infer(&typecheck::global_env(), &expr) {
            Ok(_) => {}
            Err(e) => {
                print_diagnostic(e, &line);
                continue;
            }
        }
        let result = match eval::Evaluator::new().evaluate(&expr) {
            Ok(distributions) => distributions,
            Err(e) => {
                print_diagnostic(e, &line);
                continue;
            }
        };
        match result {
            eval::RuntimeValue::Int(i) => println!("{}", i),
            eval::RuntimeValue::List(is) => println!(
                "[{}]",
                is.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            eval::RuntimeValue::Distribution(d) => {
                for distribution in d.iter() {
                    print_distribution(distribution);
                }
            }
            eval::RuntimeValue::Primitive(_) => println!("#<primitive>"),
            eval::RuntimeValue::Function(_) => println!("#<function>"),
        }
    }
}

fn print_diagnostic<T>(diagnostic: T, source_code: &str)
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
