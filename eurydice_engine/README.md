# eurydice_engine

The dice probability engine used by [Eurydice](https://eurydice.terbium.io), an
implementation of the [AnyDice](https://anydice.com/) language.

```rust
use eurydice_engine::Engine;

let mut engine = Engine::new();
let report = engine.run_with_diagnostics("output 2d6 named \"roll\"");

assert!(report.error().is_none());
assert_eq!(report.outputs[0].name, "roll");
// Outcomes are vectors of field values, paired with their probability.
assert_eq!(report.outputs[0].distribution.probabilities[0], (vec![2], 1.0 / 36.0));
```

Probabilities are computed exactly with rational arithmetic and converted to
`f64` only at the end, using the [Icepool][icepool] algorithm.[^1]

[icepool]: https://pypi.org/project/icepool/


## This project is not affiliated with AnyDice

AnyDice was created by [Jasper Flick](https://ko-fi.com/catlikecoding). This is
an independent reimplementation of its language.

## License

Apache-2.0.

[^1]: Liu, A. J. (2022). Icepool: Efficient Computation of Dice Pool
    Probabilities. _Proceedings of the AAAI Conference on Artificial
    Intelligence and Interactive Digital Entertainment, 18_(1), 258-265.
    <https://doi.org/10.1609/aiide.v18i1.21971>
