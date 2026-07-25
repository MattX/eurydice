/**
 * Wraps a WebWorker (client side), delaying communication until the webworker is ready.
 *
 * This wrapper attaches its own onMessage listener, and waits for the webworker
 * to send a message (any message will work) to signal it's ready. Before that,
 * any message or callback passed to the wrapper will be stored, and not sent / attached.
 * Once the worker is ready, the stored message will be sent, the wrapper's callback will
 * be detached, and the client's callback will be attached.
 */
export class WorkerWrapper {
  private worker: Worker;
  private workerReady = false;

  private pendingMessage?: string;
  private pendingCallback?: (event: MessageEvent) => void;

  constructor(worker: Worker) {
    this.worker = worker;
    this.worker.onmessage = () => {
      if (this.pendingCallback) {
        this.worker.onmessage = this.pendingCallback;
        this.pendingCallback = undefined;
      }
      if (this.pendingMessage !== undefined) {
        this.worker.postMessage(this.pendingMessage);
        this.pendingMessage = undefined;
      }
      this.workerReady = true;
    };
  }

  postMessage(message: string) {
    if (this.workerReady) {
      this.worker.postMessage(message);
    } else {
      this.pendingMessage = message;
    }
  }

  setOnMessage(callback: (event: MessageEvent) => void) {
    if (this.workerReady) {
      this.worker.onmessage = callback;
    } else {
      this.pendingCallback = callback;
    }
  }

  terminate() {
    this.worker.terminate();
  }
}
