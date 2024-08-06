use everydice::{
    dice::Pool,
    eval, grammar,
    output::{export_anydice_format, print_diagnostic},
};
use pretty_assertions::StrComparison;
use std::{fs, path::Path};

#[test]
fn test_anydice_programs() {
    let test_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join("anydice");
    let mut paths_with_errors = Vec::new();
    let mut ok_paths = Vec::new();
    for entry in fs::read_dir(test_dir).expect("Failed to read test directory") {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();
        let path_string = path.to_string_lossy().to_string();

        if path.is_file() {
            let content = fs::read_to_string(&path).expect("Failed to read file");
            let parts: Vec<&str> = content.split("\\\\\\ Result \\\\\\").collect();

            if parts.len() != 2 {
                paths_with_errors.push(path_string);
                println!("Invalid test file format: {}", path.display());
                continue;
            }

            let program = parts[0].trim();
            let expected_result = parts[1].trim_start().trim_end();

            let mut evaluator = eval::Evaluator::new();
            let parser = grammar::BodyParser::new();
            let mut actual_result = String::new();

            let statements = match parser.parse(program) {
                Ok(expr) => expr,
                Err(err) => {
                    paths_with_errors.push(path_string);
                    println!("Parsing error in file {}: {}", path.display(), err);
                    continue;
                }
            };

            for statement in statements {
                match evaluator.execute(&statement) {
                    Ok(()) => {}
                    Err(e) => {
                        paths_with_errors.push(path_string.clone());
                        print_diagnostic(e, program);
                        continue;
                    }
                }
            }

            for (value, name) in evaluator.get_outputs() {
                let d = match value {
                    eval::RuntimeValue::Int(i) => Pool::from_list(1, vec![i]),
                    eval::RuntimeValue::List(is) => Pool::from_list(1, is.to_vec()),
                    eval::RuntimeValue::Pool(d) => (*d).clone(),
                };
                actual_result.push_str(&export_anydice_format(&name, &d));
                actual_result.push('\n');
            }

            if actual_result.trim() != expected_result {
                paths_with_errors.push(path_string);
                println!("Mismatch in file {}:", path.display());
                println!(
                    "{}",
                    StrComparison::new(actual_result.trim(), expected_result)
                );
            } else {
                ok_paths.push(path_string);
            }
        }
    }
    println!("Paths without errors:");
    for path in &ok_paths {
        println!("✅ {}", path);
    }
    println!("Paths with errors:");
    for path in &paths_with_errors {
        println!("❌ {}", path);
    }
    assert_eq!(paths_with_errors.len(), 0, "Some test files had errors");
}
