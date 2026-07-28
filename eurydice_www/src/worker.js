import { primitiveMetadata, runWithDiagnostics } from "eurydice_wasm";

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
// Signal readiness and provide editor metadata before evaluating any program.
self.postMessage({ Ready: { primitives: primitiveMetadata() } });
