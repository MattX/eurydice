import { describe, expect, it, vi } from "vitest";

import { WorkerWrapper } from "./worker-wrapper";

function fakeWorker() {
  return {
    onmessage: null,
    postMessage: vi.fn(),
    terminate: vi.fn(),
  } as unknown as Worker;
}

describe("WorkerWrapper", () => {
  it("delivers an empty program queued before worker readiness", () => {
    const worker = fakeWorker();
    const wrapper = new WorkerWrapper(worker);

    wrapper.postMessage("");
    worker.onmessage?.(new MessageEvent("message", { data: "ready" }));

    expect(worker.postMessage).toHaveBeenCalledWith("");
  });

  it("keeps only the newest program queued before worker readiness", () => {
    const worker = fakeWorker();
    const wrapper = new WorkerWrapper(worker);

    wrapper.postMessage("output 1d6");
    wrapper.postMessage("output 2d6");
    worker.onmessage?.(new MessageEvent("message", { data: "ready" }));

    expect(worker.postMessage).toHaveBeenCalledTimes(1);
    expect(worker.postMessage).toHaveBeenCalledWith("output 2d6");
  });
});
