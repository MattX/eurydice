# Everydice documentation

This is an implementation of the AnyDice DSL.

## Getting stuff to run

The packages in this directory are:

* `eurydice_engine/`: Contains a parser and evaluator for AnyDice programs (Rust).
* `eurydice_cli/`: Contains a CLI frontend for Eurydice.
* `eurydice_wasm/`: WASM wrapper for `eurydice_engine`
* `eurydice_www/`: A Vite + React + TS app to serve as a frontend.

Build the wasm with `wasm-pack ./eurydice_wasm --target no-modules`.

There are two important symlinks in `eurydice_www/src`:

* `eurydice_wasm_bg.wasm -> ../../eurydice_wasm/pkg/eurydice_wasm_bg.wasm`
* `eurydice_wasm.js -> ../../eurydice_wasm/pkg/eurydice_wasm.js`

These allow the compiled wasm to be run in a web worker for Eurydice's web frontend.

See also notes in <https://rustwasm.github.io/wasm-bindgen/examples/wasm-in-web-worker.html>.

## Eurydice documentation (todo)

### Types

* Ints are 32-bit signed integers.
* Sequences are ordered and can have repeated values.
* Dice are distributions over sequences (possibly of length 1), and possibly weighted.

### Literals

* `123`: generates the number `123`
* `{a, b..c, d:count}`: sequence literals can contain two types of subexpressions:
  * subexpressions of type `int` are inserted into the sequence, with repetition
  * range subexpressions must evaluate to an `int` on either side, and are equivalent to listing every number
    in the range, both ends included.
  * subexpressions of type `list` will be flattened into the outer list.
  * distribution subexpressions will be summed, and a list of their possible outcomes will be flattened into the
    outer list.
  * the optional `:count` modifier will cause the item to be repeated `count` times before being inserted.
* `(expr1)d(expr2)`:
  * `d(expr2)` creates a base distribution 𝒟
    * if `expr2` is an int, this expression creates a uniform distribution over `1..expr2`.
    * if `expr2` is a list, it creates a uniform distribution over the list values.
    * if `expr2` is a distribution, `d(expr2)` is equivalent to `expr2`.
  * if `expr1` is present, 𝒟 is modified as follow:
    * if `expr1` is a list, it is summed and treated as an int.
    * if `expr1` is an int, a distrbution 𝒟×𝒟×...×𝒟 (`expr1` times) is created. If i is negative, the distribution is negated.
    * if `expr1` is a distribution, it is summed to a distribution over ints,
      then each int `i` is mapped to 𝒟×𝒟×...×𝒟 (`i` times). This will result in a distribution with uneven
      outcome sizes.
