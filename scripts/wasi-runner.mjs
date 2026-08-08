#!/usr/bin/env -S node --no-warnings
// Runs a wasm32-wasip1 binary under Node's WASI, so `cargo bench --target
// wasm32-wasip1` executes on V8 — the same engine the deployed build runs on
// in a browser. Wired up as the target runner in `.cargo/config.toml`; cargo
// invokes it as `wasi-runner.mjs <binary.wasm> [args...]`.
//
// Going through WASI rather than wasm-bindgen is what makes this work at all:
// criterion needs a monotonic clock, argv, and a directory to write its
// results into, none of which wasm32-unknown-unknown provides.
import { readFile } from "node:fs/promises";
import { mkdir } from "node:fs/promises";
import { WASI } from "node:wasi";
import path from "node:path";
import process from "node:process";

const [binary, ...args] = process.argv.slice(2);
if (!binary) {
  console.error("usage: wasi-runner.mjs <binary.wasm> [args...]");
  process.exit(2);
}

// Cargo hands us `<target>/wasm32-wasip1/<profile>/deps/<name>.wasm`, which is
// the only reliable pointer to the target directory: cwd is whichever package
// cargo is benching. Criterion's results go in a sibling of the native
// `criterion` directory rather than in it, so that a wasm run cannot clobber a
// native baseline — the two are not comparable and `--baseline` would happily
// mix them.
const targetDir = path.resolve(binary, "../../../..");
const criterionHome = path.join(targetDir, "criterion-wasm");
await mkdir(criterionHome, { recursive: true });

const wasi = new WASI({
  version: "preview1",
  // argv[0] is conventionally the program name; criterion parses the rest.
  args: [binary, ...args],
  env: { ...process.env, CRITERION_HOME: criterionHome },
  // wasip1 has no current directory: paths are matched against preopens by
  // prefix. CRITERION_HOME is absolute, so the root preopen is what makes it
  // reachable.
  preopens: { "/": "/" },
  returnOnExit: true,
});

const wasm = await WebAssembly.compile(await readFile(binary));
const instance = await WebAssembly.instantiate(wasm, wasi.getImportObject());
process.exitCode = wasi.start(instance);
