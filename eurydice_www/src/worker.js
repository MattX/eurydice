async function init_wasm_in_worker(baseUrl) {
  importScripts(new URL("./eurydice_wasm.js", baseUrl).href);
  const { run } = wasm_bindgen;
  await wasm_bindgen(new URL("./eurydice_wasm_bg.wasm", baseUrl).href);

  self.onmessage = async (event) => {
    const start = performance.now();
    var workerResult;
    try {
      workerResult = run(event.data, (value, name) => {
        self.postMessage({ Print: [value, name] });
      });
    } catch (e) {
      console.error(e);
      workerResult = { Err: { message: "Internal error", from: 0, to: 0 } };
    }
    const end = performance.now();
    self.postMessage(workerResult);
    // console.log(`Worker execution time: ${end - start}ms`);
  };
  // Signal to the main thread's WorkerWrapper that the onmessage handler
  // has been attached, and the worker is ready.
  self.postMessage("ready");
}

self.onmessage = async (event) => {
  init_wasm_in_worker(event.data);
};
