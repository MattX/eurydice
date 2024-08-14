// The worker has its own scope and no direct access to functions/objects of the
// global scope. We import the generated JS file to make `wasm_bindgen`
// available which we need to initialize our Wasm code.
importScripts('./eurydice_wasm.js');

console.log('Initializing worker')

// In the worker, we have a different struct that we want to use as in
// `index.js`.
const {run} = wasm_bindgen;

async function init_wasm_in_worker() {
    // Load the wasm file by awaiting the Promise returned by `wasm_bindgen`.
    await wasm_bindgen('./eurydice_wasm_bg.wasm');

    // Set callback to handle messages passed to the worker.
    self.onmessage = async event => {
        const start = performance.now();
        var workerResult;
        try {
            workerResult = run(event.data);
        } catch (e) {
            console.error(e);
            workerResult = { Err: { message: "Internal error", from: 0, to: 0} };
        }
        const end = performance.now();
        self.postMessage(workerResult);
        console.log(`Worker execution time: ${end - start}ms`);
    };
    // Signal to the main thread's WorkerWrapper that the onmessage handler
    // has been attached, and the worker is ready.
    self.postMessage('ready');
};

init_wasm_in_worker();
