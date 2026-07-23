# `eurydice_engine` refactoring notes

The engine is in good shape overall. A sweeping rewrite does not appear
necessary: the dice algorithm is cohesive and well tested, and recent cleanup
has already removed considerable duplication.

The following ideas are ordered by expected payoff.

## 1. Clarify weighted-distribution composition without splitting `Pool` — done

`dice::Pool` currently represents two related but distinct concepts:

- a pool of `n` identical dice; and
- an arbitrary computed distribution, conventionally represented as a pool
  with dimension 1.

Introducing a separate `WeightedDistribution` type was explored, but at the
current feature level it increased conversion and runtime-dispatch complexity
without enforcing enough additional invariants to justify the cost.

Instead:

- probability-preserving mixture normalization is centralized in
  `Pool::from_mixture`;
- both `Pool::flat_map` and function-result lifting use that implementation;
- empty component distributions are handled consistently;
- unused `Pool` APIs were removed; and
- empty-pool, zero-dimension, lifting, dynamic-scope, and additive-identity
  behavior is covered by characterization tests.

The remaining dual role is intentional for now. Reconsider a separate
distribution type only if future features require materially different
operations or invariants for dice pools and computed distributions.

Possible triggers for revisiting the split:

- bugs caused by accidentally treating a multidimensional pool as an already
  summed distribution;
- several operations that are valid for distributions but invalid for pools,
  or vice versa;
- repeated dimension-one assertions or conversions across module boundaries;
- a public engine API that needs to expose computed distributions independently
  of dice pools; or
- an implementation where the distinction removes more branching and
  conversion code than it introduces.

## 2. Encapsulate runtime values and their element-type invariant — completed

Validated `SequenceValue` and `PoolValue` wrappers were implemented and then
reverted. They made invalid type/data pairs harder to construct, but added
substantial code and match-site churn without reducing defensive branches or
meaningfully simplifying operators and primitives. Much of the existing type
plumbing merely moved behind wrapper accessors.

Keep the current `RuntimeValue::List` and `RuntimeValue::Pool` representation
for now. Reconsider encapsulation only if concrete invariant bugs appear or a
future value/module refactor provides wrappers with enough behavior to remove
more code than they introduce.

## 3. Slim down `eval.rs`

`eval.rs` currently owns several distinct systems:

- runtime values and element types;
- environments and functions;
- statement and expression interpretation;
- operator semantics and distribution lifting;
- dice construction; and
- runtime diagnostics.

After establishing clearer runtime-value boundaries, these could be separated
into modules such as:

```text
value.rs
environment.rs
operators.rs
error.rs
eval.rs
```

Splitting the existing file without first improving the boundaries would
mostly move complexity around, so this should follow the first two changes.

### Environment representation

`ValEnv` uses an `Rc<RefCell<_>>` parent chain, although user functions do not
retain closure environments. If the intended AnyDice-compatible behavior is
confirmed to be dynamic scoping, an explicit stack of frames may express the
semantics more directly.

This could make function scope, block scope, recursion, and mutation easier to
reason about. It should only be attempted with focused dynamic-scope and
assignment tests in place.

## 4. Add a high-level engine API

The CLI and WASM frontends currently repeat the same workflow:

1. construct the generated parser;
2. parse source text;
3. execute statements individually;
4. translate parse and runtime errors;
5. drain outputs; and
6. convert runtime values into output distributions.

A façade such as `Engine::run(source)` could own this workflow and expose a
stable public API, while keeping the parser, AST, evaluator internals, and
runtime values available only where needed.

Expected benefits:

- less duplicated frontend glue;
- consistent diagnostics across frontends;
- a smaller supported public surface; and
- easier integration testing.

This is smaller than the first two refactors and could be done independently.

## 5. Consolidate binary-operator broadcasting

Three functions in `eval.rs` independently reimplement the same
"broadcast a binary operation across element / sequence / pool operands" matrix:

- `lift_math_binary_op` (arithmetic);
- `comp_binary_op` (ordering comparisons); and
- `equality_binary_op` (equality).

Each handles the same cases — `(element, element)` to a scalar, `(list, list)`,
`(list, element)` / `(element, list)` summed elementwise, and the `(pool, …)`
cross product folded into a new pool. `comp_binary_op` and `equality_binary_op`
differ only in the per-element function and one special case (`list == list`
compares whole sequences). The pool branch uses the same kind of
probability-preserving composition now centralized in `Pool::from_mixture`, but
still repeats the operand-shape broadcasting around it.

A single `broadcast_binary` helper, parameterized by the per-element function
and the list/list rule, would collapse all three. This would naturally live in
the `operators.rs` extracted in §3, so it is best done alongside that work, not
before.

Expected benefits:

- one implementation of operand broadcasting instead of three; and
- less risk of the variants drifting apart during future changes.

## 6. Reduce primitive declaration boilerplate

`primitives.rs` declares each built-in twice: once as a `pub static
_PRIMITIVE` value (`arg_types`, `accepts_non_numeric`, `execute`) and again as a
name string in the `register_primitives` table. The name/executor pairing is
therefore split across two locations and guarded only by a hard-coded
`assert_eq!(functions.len(), 20)` in the tests.

Several executors are also near-duplicates: `highest_of_execute` /
`lowest_of_execute` differ by one method call, and `contains_execute` /
`count_execute` share the whole "merge outcome types, materialize identities,
re-destructure" preamble.

A single declarative table (or a small macro) pairing name, argument types, and
executor in one place would remove the split registration and the count
assertion, and a shared helper for the outcome-type-merging preamble would
absorb the duplicated executor bodies. This is independent of the abstractions
above and low-risk, so it can be done at any time.

Expected benefits:

- name, types, and behavior for a primitive defined in one place; and
- less copy-paste across executors with identical type-handling preambles.

## What not to refactor yet

The Icepool recursion and multiset iterators in `dice.rs` are relatively
self-contained, documented, and well covered. They do not currently justify a
large algorithmic rewrite.

Similarly, breaking large files into smaller files is not valuable by itself.
Module extraction should follow clearer ownership and invariants.

## Suggested order

1. Characterization coverage and shared mixture normalization — completed.
2. Evaluate encapsulating runtime element-type invariants — completed; the
   wrapper approach was reverted.
3. Move operator and value behavior out of `eval.rs`.
4. Reassess whether the resulting module boundaries naturally justify distinct
   pool and distribution types.
5. Consider replacing the environment parent chain with explicit frames.
6. Add the high-level `Engine` façade and migrate the CLI and WASM frontends.

The primitive-table cleanup (§6) is independent of the above and can be done at
any point; consolidating operator broadcasting (§5) is best folded into step 3.

## Baseline when these notes were written

- `cargo check -p eurydice_engine` passes.
- All 97 `eurydice_engine` tests pass.
- Strict Clippy reports nine minor warnings, almost entirely casts in tests;
  these do not indicate an architectural problem.
