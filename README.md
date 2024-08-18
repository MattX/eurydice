# Everydice documentation

This is an implementation of the [AnyDice](https://anydice.com/) DSL, created by [Jasper Flick](https://ko-fi.com/catlikecoding). **This project is not affiliated with AnyDice or Jasper Flick.**

It can run as a command-line utility, or as a web-based tool, with evaluation occurring in real time in the browser.

## Running

See the web UI at xxx.

You can also run a simple CLI locally. [Install cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html) if necessary, then:

```sh
git clone https://github.com/MattX/eurydice.git
cd eurydice/eurydice_cli
cargo run --release
```

Try:

```
> output 2d3 + d4
output 1:
   3    2.778% |━━━━━━━━
   4    8.333% |━━━━━━━━━━━━━━━━━━━━━━━━
   5   16.667% |━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   6   22.222% |━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   7   22.222% |━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   8   16.667% |━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   9    8.333% |━━━━━━━━━━━━━━━━━━━━━━━━
  10    2.778% |━━━━━━━━
```

## Documentation

See the [language specification](/spec.md), or the [AnyDice docs](https://anydice.com/docs/).

## Contributing

Contributions are more than welcome! Either in the UI or the core language. Some ideas:

* New output options
* Many performance optimizations could be made
* More test coverage is always welcome

### Getting stuff to run

The packages in this directory are:

* `eurydice_engine`: Contains a parser and evaluator for AnyDice programs (Rust).
* `eurydice_cli`: Contains a CLI frontend for Eurydice. **Most serious tests are in this package** so we can display results in a pretty way without bundling terminal dependencies in `eurydice_engine`.
* `eurydice_wasm`: WASM wrapper for `eurydice_engine`
* `eurydice_www`: A Vite + React + TS app to serve as a frontend.

Build the wasm (from this directory) with `wasm-pack build ./eurydice_wasm --target no-modules`.

There are two important symlinks in `eurydice_www/src`:

* `eurydice_wasm_bg.wasm -> ../../eurydice_wasm/pkg/eurydice_wasm_bg.wasm`
* `eurydice_wasm.js -> ../../eurydice_wasm/pkg/eurydice_wasm.js`

These allow the compiled wasm to be run in a web worker for Eurydice's web frontend.

See also notes in <https://rustwasm.github.io/wasm-bindgen/examples/wasm-in-web-worker.html>.
