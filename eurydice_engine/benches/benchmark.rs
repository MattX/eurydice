use criterion::{Criterion, criterion_group, criterion_main};
use eurydice_engine::{
    ast::{Statement, WithRange},
    eval::Evaluator,
};

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

fn criterion_benchmark(c: &mut Criterion) {
    for (name, source) in PROGRAMS {
        let parsed = eurydice_engine::grammar::BodyParser::new()
            .parse(source)
            .unwrap_or_else(|error| panic!("`{name}` failed to parse: {error:?}"));
        c.bench_function(name, |b| {
            b.iter(|| execute_all(&mut Evaluator::new(), &parsed))
        });
    }
}

fn execute_all(eval: &mut Evaluator, parsed: &[WithRange<Statement>]) {
    for stmt in parsed {
        eval.execute(stmt).unwrap();
    }
}

/// Every benchmarked program, as `(name, source)`.
///
/// The engine's time goes into whole-container work: summing a pool through
/// the Icepool state machine, building and mapping long sequences, and
/// evaluating a function once per multiset of a pool. Each case here drives
/// one of those paths at a size where a regression shows, and the comment on
/// each names what it is watching.
static PROGRAMS: &[(&str, &str)] = &[
    ("2x3d10", ITERATIONS),
    ("looped math", LOOPED_MATH),
    ("sum pool wide", SUM_POOL_WIDE),
    ("sum pool keep", SUM_POOL_KEEP),
    ("sum pool tuples", SUM_POOL_TUPLES),
    ("sequence literal long", SEQUENCE_LITERAL_LONG),
    ("sequence broadcast", SEQUENCE_BROADCAST),
    ("sequence sort", SEQUENCE_SORT),
    ("function over pool", FUNCTION_OVER_POOL),
    ("pool comparison", POOL_COMPARISON),
    ("symbol pool count", SYMBOL_POOL_COUNT),
    ("explode pool", EXPLODE_POOL),
];

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

/// Summing a pool with no keep list: the plain path through the Icepool state
/// machine, over enough dice that the memoized recursion does real work.
const SUM_POOL_WIDE: &str = "output 8d20";

/// The same state machine reached through a keep list, which is a separate
/// entry point and folds a different number of dice per outcome.
const SUM_POOL_KEEP: &str = "output [highest 3 of 10d20]";

/// Summing tuple-valued faces, where combining two outcomes recurses field by
/// field instead of adding two integers.
const SUM_POOL_TUPLES: &str = "
    SIDES: {
        [tuple 1 2], [tuple 3 4], [tuple 5 6], [tuple 7 8],
        [tuple 2 1], [tuple 4 3], [tuple 6 5], [tuple 8 7]
    }
    output 8dSIDES
";

/// Building one long sequence and rendering it, which touches every element
/// twice: once on construction and once on display.
const SEQUENCE_LITERAL_LONG: &str = "output {1..20000}";

/// Applying an element operation across a long sequence, then summing the
/// results back into a single value.
const SEQUENCE_BROADCAST: &str = "
    X: {1..20000}
    output X + 1
";

/// Reordering a long sequence. `reverse` first so `sort` gets its worst case
/// rather than an already-ordered input.
const SEQUENCE_SORT: &str = "output [sort [reverse {1..20000}]]";

/// Calling a function once per multiset of a pool and combining the results,
/// which is the widest fan-out the evaluator has.
const FUNCTION_OVER_POOL: &str = "
    function: f A:s {
        result: [sum integers in A]
    }
    output [f 6d10]
";

/// Comparing two pools, which sums both and then walks their cross product.
const POOL_COMPARISON: &str = "output 5d10 < 5d10";

/// Counting over a pool whose outcomes are symbols: the path that cannot be
/// summed but is not an error, so it stays worth measuring on its own.
const SYMBOL_POOL_COUNT: &str = "
    enum: RESULT { A, B, C, D, E, F }
    output [count {A} in 12d{A, B, C, D, E, F}]
";

/// A transform primitive, which rebuilds the pool once per explosion depth.
const EXPLODE_POOL: &str = "output [explode 3d10]";
