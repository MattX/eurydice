//! Helpers shared by the integration tests.
//!
//! Everything here goes through [`Engine`], the same entry point a frontend
//! uses. A harness that assembles outputs its own way can only test its own
//! assembly, so these run programs exactly as a caller of the crate would.

// Each test binary compiles this module separately and uses only part of it.
#![allow(dead_code)]

use eurydice_engine::{Distribution, Engine, EngineDiagnostic};

/// Runs a program, returning one distribution per `output` statement, or the
/// diagnostic that stopped it.
pub fn run(program: &str) -> Result<Vec<Distribution>, Box<EngineDiagnostic>> {
    let report = Engine::new().run_source(program);
    match report.error() {
        Some(error) => Err(Box::new(error.clone())),
        None => Ok(report
            .outputs
            .into_iter()
            .map(|output| output.distribution)
            .collect()),
    }
}

/// The probabilities of every output of a program that is expected to run.
pub fn probabilities(program: &str) -> Vec<Vec<(Vec<i32>, f64)>> {
    run(program)
        .unwrap_or_else(|error| panic!("{program}: {}", error.summary))
        .into_iter()
        .map(|distribution| distribution.probabilities)
        .collect()
}

/// The probabilities of the single output of a program that is expected to run.
pub fn only_probabilities(program: &str) -> Vec<(Vec<i32>, f64)> {
    let mut outputs = probabilities(program);
    assert_eq!(outputs.len(), 1, "{program}: expected exactly one output");
    outputs.remove(0)
}

/// The distributions of a program that is expected to run.
pub fn distributions(program: &str) -> Vec<Distribution> {
    run(program).unwrap_or_else(|error| panic!("{program}: {}", error.summary))
}

/// The diagnostic a failing program produces.
pub fn error(program: &str) -> EngineDiagnostic {
    match run(program) {
        Err(error) => *error,
        Ok(_) => panic!("expected an error for {program}"),
    }
}

/// The diagnostic a program produces, or `None` if it succeeded.
pub fn diagnostic(program: &str) -> Option<EngineDiagnostic> {
    run(program).err().map(|error| *error)
}

/// The message a failing program shows the user.
pub fn error_summary(program: &str) -> String {
    error(program).summary
}
