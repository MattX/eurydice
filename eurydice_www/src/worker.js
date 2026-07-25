import { run } from "eurydice_wasm";

self.onmessage = (event) => {
  try {
    const result = run(event.data, (value, name) => {
      self.postMessage({ Print: [value, name] });
    });
    self.postMessage(result);
  } catch (e) {
    console.error(e);
    self.postMessage({
      Err: { message: "Internal error", from: 0, to: 0 },
    });
  }
};
// Signal to the main thread's WorkerWrapper that the onmessage handler
// has been attached, and the worker is ready.
self.postMessage("ready");
