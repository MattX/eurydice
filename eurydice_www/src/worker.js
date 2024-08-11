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
        var worker_result = run(event.data);
        const end = performance.now();
        self.postMessage(worker_result);
        console.log(`Worker execution time: ${end - start}`);
    };
};

init_wasm_in_worker();
