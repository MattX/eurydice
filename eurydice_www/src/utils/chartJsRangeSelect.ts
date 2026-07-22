import { Chart, Plugin } from "chart.js";

export interface ChartJsRangeSelectOptions {
  // Called when the range is changed.
  onRangeChange?: (startValue: number, endValue: number) => void;
  // Called when the user releases the mouse button.
  onDragEnd?: (startValue: number, endValue: number) => void;
}

export interface ChartJsRangeSelect {
  // Toggle whether user can interact with the plugin.
  setEnabled: (enabled: boolean) => void;
  // Clear the range and stop showing the range selection.
  setActive: (active: boolean) => void;
  // Set the range to be shown.
  setRange: (startValue: number, endValue: number) => void;
  // Set the value associated with tick 0.
  setOffset: (offset: number) => void;
  // The plugin to be used in the chart.js chart.
  plugin: Plugin;
}

export function makeChartJsRangeSelect(
  options: ChartJsRangeSelectOptions
): ChartJsRangeSelect {
  let enabled = true; // Controls interaction
  let active = false; // Controls visual display
  let startValue = 0;
  let endValue = 0;
  let offset = 0;
  let chart: Chart | null = null;

  let isDragging = false;

  const updateChart = () => {
    // React can detach the canvas before react-chartjs-2 destroys the chart.
    // Updating during that gap makes Chart.js inspect a null parent element.
    if (chart?.canvas.isConnected) {
      chart.update();
    }
  };

  const setEnabled = (newEnabled: boolean) => {
    enabled = newEnabled;
  };

  const setActive = (newActive: boolean) => {
    if (isDragging) {
      return;
    }

    active = newActive;
    updateChart();
  };

  const setRange = (newStartValue: number, newEndValue: number) => {
    startValue = newStartValue - offset;
    endValue = newEndValue - offset;
    updateChart();
  };

  const setOffset = (newOffset: number) => {
    startValue = startValue + offset - newOffset;
    endValue = endValue + offset - newOffset;
    offset = newOffset;
    updateChart();
  };

  const mouseDown = (event: MouseEvent) => {
    if (chart === null || !enabled) {
      return;
    }

    isDragging = true;
    active = true;
    startValue = chart.scales["x"].getValueForPixel(event.offsetX) ?? 0;
    endValue = startValue;
    options.onRangeChange?.(
      startValue + offset,
      endValue + offset
    );
  };

  const mouseUp = () => {
    if (isDragging) {
      options.onDragEnd?.(startValue + offset, endValue + offset);
    }
    isDragging = false;
  };

  const mouseMove = (event: MouseEvent) => {
    if (chart === null || !enabled || !isDragging) {
      return;
    }

    endValue = chart.scales["x"].getValueForPixel(event.offsetX) ?? 0;
    options.onRangeChange?.(
      startValue + offset,
      endValue + offset
    );
  };

  const plugin: Plugin = {
    id: "range-select",

    start: (newChart) => {
      chart = newChart;
      chart.canvas.addEventListener("mousedown", mouseDown);
      chart.canvas.addEventListener("mouseup", mouseUp);
      chart.canvas.addEventListener("mouseleave", mouseUp);
      chart.canvas.addEventListener("mousemove", mouseMove);
    },

    stop: (stoppedChart) => {
      stoppedChart.canvas?.removeEventListener("mousedown", mouseDown);
      stoppedChart.canvas?.removeEventListener("mouseup", mouseUp);
      stoppedChart.canvas?.removeEventListener("mouseleave", mouseUp);
      stoppedChart.canvas?.removeEventListener("mousemove", mouseMove);
      if (chart === stoppedChart) {
        chart = null;
        isDragging = false;
      }
    },

    beforeDraw: (chart) => {
      if (!active) {
        return;
      }

      const ctx = chart.ctx;
      ctx.fillStyle = "rgba(0, 0, 0, 0.2)";
      const startX = chart.scales["x"].getPixelForValue(startValue);
      const endX = chart.scales["x"].getPixelForValue(endValue);
      ctx.fillRect(
        startX,
        chart.chartArea.top,
        endX - startX,
        chart.chartArea.height
      );
    },
  };

  return {
    setEnabled,
    setActive,
    setRange,
    setOffset,
    plugin,
  };
}
