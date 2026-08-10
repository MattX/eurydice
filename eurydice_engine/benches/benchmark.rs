use criterion::{Criterion, criterion_group, criterion_main};
use eurydice_engine::Engine;

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

fn criterion_benchmark(c: &mut Criterion) {
    for (name, source) in PROGRAMS {
        // Each program is parsed once, outside the measured closure: parsing
        // costs a quarter to a third of a short pool program's total time,
        // which would swamp the evaluation signal these benchmarks watch.
        let program = Engine::new()
            .compile(source)
            .unwrap_or_else(|report| panic!("`{name}` failed to parse: {:?}", report.error()));
        c.bench_function(name, |b| {
            // A fresh engine per iteration: definitions and settings persist
            // across runs, so a shared one would not be measuring the same work
            // twice.
            b.iter(|| std::hint::black_box(Engine::new().run(&program)))
        });
    }
}

/// Every benchmarked program, as `(name, source)`.
///
/// The engine's time goes into whole-container work: summing a pool through the
/// Icepool state machine, evaluating a function once per multiset of a pool,
/// building a pool from a mixture of dice, laying out each output for display,
/// and mapping long sequences. Each case here drives one of those paths at a
/// size where a regression shows, and the comment on each names what it is
/// watching.
///
/// Most of these are real programs rather than constructed ones: where a case
/// is drawn from `tests/fixtures/`, the comment cites the fixture. Keeping the
/// two corpora recognisably the same body of programs means a fixture that
/// starts failing and a benchmark that starts regressing point at the same
/// code.
///
/// This bench also runs under wasm — see `scripts/wasi-runner.mjs` and the
/// README. Nothing here may depend on the filesystem or on threads, which is
/// why the programs are inline rather than loaded from a directory.
static PROGRAMS: &[(&str, &str)] = &[
    ("sum pool wide", SUM_POOL_WIDE),
    ("sum pool keep", SUM_POOL_KEEP),
    ("pool wide dimension", POOL_WIDE_DIMENSION),
    ("pool comparison", POOL_COMPARISON),
    ("function over pool", FUNCTION_OVER_POOL),
    ("success pool", SUCCESS_POOL),
    ("explode and keep", EXPLODE_AND_KEEP),
    ("recursive die", RECURSIVE_DIE),
    ("attack pipeline", ATTACK_PIPELINE),
    ("output sweep", OUTPUT_SWEEP),
    ("symbol pool", SYMBOL_POOL),
    ("tuples", TUPLES),
    ("sequence sort", SEQUENCE_SORT),
    ("scalar loop", SCALAR_LOOP),
];

// --- Pool summation: the Icepool state machine ------------------------------

/// Summing a pool with no keep list: the plain path through the Icepool state
/// machine, over few dice with many outcomes each.
const SUM_POOL_WIDE: &str = "output 8d20";

/// The same state machine reached through a keep list, which is a separate
/// entry point and folds a different number of dice per outcome. This is also
/// the most common idiom in real programs — advantage, 4d6-drop-lowest.
const SUM_POOL_KEEP: &str = "output [highest 3 of 10d20]";

/// The complement of `sum pool wide`: many dice with two outcomes each, which
/// drives the `0..=dimension` loop in `apply_inner` at its longest rather than
/// its widest. From `eurydice_www/src/samples/catan.txt`.
const POOL_WIDE_DIMENSION: &str = "output 60d(2d6 = 7)";

/// Comparing two pools, which sums both and then walks their cross product.
const POOL_COMPARISON: &str = "output 5d10 < 5d10";

// --- Fan-out and recursion --------------------------------------------------

/// Calling a function once per multiset of a pool and combining the results,
/// which is the widest fan-out the evaluator has. Two pool arguments means the
/// cross product of both multiset enumerations.
const FUNCTION_OVER_POOL: &str = "
    function: iterate A:s B:s {
        result: A < B
    }
    output [iterate 3d10 3d10]
";

/// Count-successes with a botch rule, the dice-pool-game workload: the same
/// fan-out as `function over pool` but in the shape users write it, and once
/// per pool size. From `tests/fixtures/anydice/blog_exalted_2.txt`, with the
/// loop scaled from `{1..10}` down to `{1..6}` to keep the case affordable.
const SUCCESS_POOL: &str = r#"
    function: evaluate ROLL:s {
        SUCCESSES: ROLL >= 7
        if SUCCESSES > 0 { result: SUCCESSES + (ROLL = 10) }
        if [count 1 in ROLL] { result: -1 }
        result: 0
    }

    loop DICE over {1..6} {
        output [evaluate DICE d10] named "[DICE]d"
    }
"#;

/// Roll-and-keep with exploding dice: a transform primitive rebuilding the pool
/// once per explosion depth, feeding a keep list. From
/// `tests/fixtures/anydice/l5r.txt`, verbatim.
const EXPLODE_AND_KEEP: &str = r#"
    set "explode depth" to 4
    output [highest 3 of 5d[explode d10]]
"#;

/// A die defined by a recursive function, then rolled as a pool. The only case
/// that reaches the `maximum function depth` cutoff. From
/// `tests/fixtures/anydice/blog_nwod.txt`, with the loop scaled to `{1..6}`.
const RECURSIVE_DIE: &str = r#"
    function: nwod R:n again N:n {
        if N >= R { result: 1 + [nwod R again d10] }
        result: N >= 8
    }

    NWOD: [nwod 10 again d10]

    loop N over {1..6} {
        output [lowest of 10 and NdNWOD] named "[N]d"
    }
"#;

// --- Realistic whole programs -----------------------------------------------

/// A full attack-and-damage pipeline. `2d[attack d20 vs DEFENSE]` builds a pool
/// from a mixture of dice, which rescales weights with `Lcm`/`DivExact` on
/// arbitrary-precision integers — a path nothing else here touches. Also
/// exercises `@`, `contains`, `maximum of`, and a loop inside a function
/// evaluated over a pool. From
/// `tests/fixtures/anydice/two_fanged_strike.txt`, verbatim.
const ATTACK_PIPELINE: &str = r#"
    MISS: 0
    HIT: 1
    CRITICALHIT: 2

    function: attack ROLL:n vs DEFENSE:n {
        if ROLL = 1 { result: MISS }
        if ROLL = 20 {
            if ROLL + ATTACK >= DEFENSE { result: CRITICALHIT }
            result: HIT
        }
        if ROLL + ATTACK >= DEFENSE { result: HIT }
        result: MISS
    }

    function: damage for ATTACKS:s {
        D: 0
        loop A over {1..#ATTACKS} {
            if A@ATTACKS = CRITICALHIT { D: D + CRITICAL }
            else if A@ATTACKS = HIT { D: D + DAMAGE }
        }
        result: D + [extra damage for ATTACKS]
    }

    function: extra damage for ATTACKS:s {
        if [ATTACKS contains CRITICALHIT] { D: [maximum of QUARRY] }
        else if [ATTACKS contains HIT] { D: QUARRY }
        else { D: 0 }
        if ![ATTACKS contains MISS] {
            result: D + 4
        }
        result: D
    }

    function: twin strike {
        result: [damage for 2d[attack d20 vs DEFENSE]]
    }

    DEFENSE: d{12..22}
    QUARRY: d6
    ATTACK: 6 + 2
    DAMAGE: d12 + 6
    CRITICAL: 18
    output [twin strike] named "Greatbow"
"#;

/// Nineteen outputs from one program. Everything else here renders one or two,
/// so this is what puts weight on the per-output work — summing each stored
/// value, discovering its field schema, and laying out its outcomes. Sweeping a
/// target number and graphing the result is a common shape. From the second
/// half of `tests/fixtures/anydice/blog_dnd_3.txt`.
const OUTPUT_SWEEP: &str = r#"
    MISS: 0
    NORMALHIT: 1
    CRITICALHIT: 2

    ATTACK: 0
    THREAT: 15

    function: ROLL:n vs AC:n {
        if ROLL = 1 { result: MISS }
        if ROLL >= THREAT & ROLL + ATTACK >= AC {
            result: [threat d20]
        }
        if ROLL + ATTACK >= AC | ROLL = 20 { result: NORMALHIT }
        result: MISS
    }

    function: threat ROLL:n {
        if ROLL = 1 { result: NORMALHIT }
        if ROLL + ATTACK >= AC | ROLL = 20 {
            result: CRITICALHIT
        }
        result: NORMALHIT
    }

    loop AC over {2..20} {
        output [d20 vs AC] named "[AC]"
    }
"#;

// --- Eurydice-only outcome types --------------------------------------------

/// A pool whose outcomes are symbols cannot be summed, but `count` and
/// `contains` take the pool itself and so go through Icepool rather than
/// enumerating it. This pool has 888030 multisets, which is the point: the
/// case is fast only if that routing holds. From
/// `tests/fixtures/eurydice/enum_pool_multisets.txt`.
const SYMBOL_POOL: &str = r#"
    enum { FA, FB, FC, FD, FE, FF, FG, FH }
    LARGE: 20d{FA, FB, FC, FD, FE, FF, FG, FH}
    output [count {FA} in LARGE] named "count"
    output [LARGE contains FA] named "contains"
"#;

/// Both tuple paths in one program. `8dSIDES` sums tuple-valued faces, where
/// combining two outcomes recurses field by field instead of adding two
/// integers. The Risk round instead builds tuples inside a function evaluated
/// over two pools and mixes the results. Second half from
/// `tests/fixtures/eurydice/tuple_distributions.txt`.
const TUPLES: &str = r#"
    SIDES: {
        [tuple 1 2], [tuple 3 4], [tuple 5 6], [tuple 7 8],
        [tuple 2 1], [tuple 4 3], [tuple 6 5], [tuple 8 7]
    }
    output 8dSIDES named "tuple faces"

    function: risk round ATTACK:s versus DEFENCE:s {
        COMPARISONS: [lowest of #ATTACK and #DEFENCE]
        ATTACKER_LOSSES: 0
        DEFENDER_LOSSES: 0
        loop I over {1..COMPARISONS} {
            if I@ATTACK > I@DEFENCE {
                DEFENDER_LOSSES: DEFENDER_LOSSES + 1
            }
            else {
                ATTACKER_LOSSES: ATTACKER_LOSSES + 1
            }
        }
        result: [tuple ATTACKER_LOSSES DEFENDER_LOSSES]
    }
    output [risk round 3d6 versus 2d6] named "risk round"
"#;

// --- Micro-benchmarks, deliberately -----------------------------------------

/// The one long-sequence case: builds 20000 elements, reverses them so `sort`
/// gets its worst case rather than an already-ordered input, sorts, and renders
/// the result. No user writes this, but it is what catches an accidentally
/// quadratic list operation.
const SEQUENCE_SORT: &str = "output [sort [reverse {1..20000}]]";

/// 20000 iterations of scalar arithmetic. Nothing realistic about it either; it
/// is the only case that isolates statement dispatch and environment lookup
/// from distribution math, so a regression there shows here and nowhere else.
const SCALAR_LOOP: &str = "
    TOTAL: 0
    loop X over {1..200} {
        loop Y over {1..100} {
            TOTAL: TOTAL + X / 2 - Y * 3
        }
    }

    output TOTAL
";
