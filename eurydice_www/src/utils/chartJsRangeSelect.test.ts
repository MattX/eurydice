import { Chart } from "chart.js";
import { describe, expect, it, vi } from "vitest";
import { makeChartJsRangeSelect } from "./chartJsRangeSelect";

describe("chartJsRangeSelect lifecycle", () => {
  it("does not update a detached or stopped chart", () => {
    const canvas = {
      isConnected: true,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    };
    const update = vi.fn();
    const chart = { canvas, update } as unknown as Chart;
    const rangeSelect = makeChartJsRangeSelect({});

    (rangeSelect.plugin.start as (chart: Chart) => void)(chart);
    rangeSelect.setActive(true);
    expect(update).toHaveBeenCalledTimes(1);

    canvas.isConnected = false;
    expect(() => rangeSelect.setRange(1, 2)).not.toThrow();
    expect(update).toHaveBeenCalledTimes(1);

    (rangeSelect.plugin.stop as (chart: Chart) => void)(chart);
    canvas.isConnected = true;
    rangeSelect.setActive(false);
    expect(update).toHaveBeenCalledTimes(1);
    expect(canvas.removeEventListener).toHaveBeenCalledTimes(4);
  });
});
