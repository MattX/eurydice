mod ast;
mod eval;
mod output;
// mod parse;
mod probability;
mod typecheck;

use std::io::{self, BufRead as _};

use lalrpop_util::lalrpop_mod;
use miette::{Diagnostic, GraphicalReportHandler};
use output::print_distribution;

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
        match typecheck::infer(&typecheck::global_env(), &expr) {
            Ok(_) => {}
            Err(e) => {
                print_diagnostic(e, &line);
                continue;
            }
        }
        let result = match eval::Evaluator::new(&line).evaluate(&expr) {
            Ok(distributions) => distributions,
            Err(e) => {
                print_diagnostic(e, &line);
                continue;
            }
        };
        for distribution in result {
            print_distribution(&distribution);
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
where T: std::error::Error, T: Diagnostic {
    source: T,
    source_code: U,
}

impl<T, U> std::fmt::Display for DiagnosticSourceAdapter<T, U>
where
    T: std::error::Error, T:Diagnostic,
    U: miette::SourceCode,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.source)
    }
}

impl<T, U> std::fmt::Debug for DiagnosticSourceAdapter<T, U>
where
    T: std::error::Error, T:Diagnostic,
    U: miette::SourceCode,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.source)
    }
}

impl<T, U> std::error::Error for DiagnosticSourceAdapter<T, U>
where
    T: std::error::Error, T:Diagnostic,
    U: miette::SourceCode,
{
}

impl<T, U> Diagnostic for DiagnosticSourceAdapter<T, U>
where
    T: std::error::Error, T:Diagnostic,
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
