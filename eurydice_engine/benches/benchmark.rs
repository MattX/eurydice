use criterion::{criterion_group, criterion_main, Criterion};
use eurydice_engine::{
    ast::{Statement, WithRange},
    eval::Evaluator,
};

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

fn criterion_benchmark(c: &mut Criterion) {
    let parsed = eurydice_engine::grammar::BodyParser::new()
        .parse(ITERATIONS)
        .unwrap();
    c.bench_function("2x3d10", |b| {
        b.iter(|| execute_all(&mut Evaluator::new(), &parsed))
    });

    let parsed = eurydice_engine::grammar::BodyParser::new()
        .parse(LOOPED_MATH)
        .unwrap();
    c.bench_function("looped math", |b| {
        b.iter(|| execute_all(&mut Evaluator::new(), &parsed))
    });
}

fn execute_all(eval: &mut Evaluator, parsed: &[WithRange<Statement>]) {
    for stmt in parsed {
        eval.execute(stmt).unwrap();
    }
}

const ITERATIONS: &str = "
    function: iterate A:s B:s {
        result: A < B
    }
    output [iterate 3d10 3d10]
";

const LOOPED_MATH: &str = "
    TOTAL: 0
    loop X over {1..1000} {
        loop Y over {1..100} {
            TOTAL: TOTAL + X / 2 - Y * 3
        }
    }

    output TOTAL
";
