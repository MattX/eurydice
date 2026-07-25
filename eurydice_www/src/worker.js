import { runWithDiagnostics } from "eurydice_wasm";

self.onmessage = (event) => {
  try {
    const report = runWithDiagnostics(event.data, (value, name) => {
      self.postMessage({ Print: [value, name] });
    });
    self.postMessage({ Report: report });
  } catch (e) {
    console.error(e);
    self.postMessage({
      InternalError: "The evaluation worker encountered an internal error.",
    });
  }
};
// Signal to the main thread's WorkerWrapper that the onmessage handler
// has been attached, and the worker is ready.
self.postMessage("ready");
