/**
 * Wraps a WebWorker (client side), delaying communication until the webworker is ready.
 *
 * The worker signals readiness by sending its first message. Until then, a
 * message passed to the wrapper is held back rather than posted. Events that
 * arrive before the client attaches a callback — starting with the readiness
 * event itself — are queued and replayed in order once it does, so no message
 * is lost to a late listener.
 */
export class WorkerWrapper {
  private worker: Worker;
  private workerReady = false;

  private pendingMessage?: string;
  private callback?: (event: MessageEvent) => void;
  private pendingEvents: MessageEvent[] = [];

  constructor(worker: Worker) {
    this.worker = worker;
    this.worker.onmessage = (event) => {
      const wasReady = this.workerReady;
      this.workerReady = true;
      this.deliver(event);
      if (!wasReady) {
        this.sendPendingMessage();
      }
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
    this.callback = callback;
    const queued = this.pendingEvents;
    this.pendingEvents = [];
    for (const event of queued) {
      callback(event);
    }
  }

  private deliver(event: MessageEvent) {
    if (this.callback) {
      this.callback(event);
    } else {
      this.pendingEvents.push(event);
    }
  }

  private sendPendingMessage() {
    if (this.pendingMessage !== undefined) {
      this.worker.postMessage(this.pendingMessage);
      this.pendingMessage = undefined;
    }
  }

  terminate() {
    this.worker.terminate();
  }
}
