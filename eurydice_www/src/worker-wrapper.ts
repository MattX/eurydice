/**
 * Wraps a WebWorker (client side), delaying communication until the webworker is ready.
 *
 * This wrapper attaches its own onMessage listener, and waits for the webworker
 * to send a message to signal it's ready. Before that,
 * any message or callback passed to the wrapper will be stored, and not sent / attached.
 * Once the worker is ready, its readiness event is forwarded to the client's callback
 * and the stored program is sent.
 */
export class WorkerWrapper {
  private worker: Worker;
  private workerReady = false;

  private pendingMessage?: string;
  private pendingCallback?: (event: MessageEvent) => void;
  private readyEvent?: MessageEvent;
  private pendingEvents: MessageEvent[] = [];

  constructor(worker: Worker) {
    this.worker = worker;
    this.worker.onmessage = (event) => {
      this.workerReady = true;
      const callback = this.pendingCallback;
      if (callback) {
        this.worker.onmessage = callback;
        this.pendingCallback = undefined;
        callback(event);
      } else {
        this.readyEvent = event;
        this.worker.onmessage = (pendingEvent) => {
          this.pendingEvents.push(pendingEvent);
        };
      }
      this.sendPendingMessage();
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
      if (this.readyEvent) {
        const event = this.readyEvent;
        this.readyEvent = undefined;
        callback(event);
      }
      for (const event of this.pendingEvents) {
        callback(event);
      }
      this.pendingEvents = [];
    } else {
      this.pendingCallback = callback;
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
