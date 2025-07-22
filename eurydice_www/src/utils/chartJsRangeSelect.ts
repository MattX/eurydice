import { Chart, Plugin } from "chart.js";

export interface ChartJsRangeSelectOptions {
    // Called when the range is changed.
    onRangeChange?: (startValue: number, endValue: number) => void;
}

export interface ChartJsRangeSelect {
    // Clear the range and stop showing the range selection.
    setEnabled: (enabled: boolean) => void;
    // Set the range to be shown.
    setRange: (startValue: number, endValue: number) => void;
    // The plugin to be used in the chart.js chart.
    plugin: Plugin;
}

export function makeChartJsRangeSelect(options: ChartJsRangeSelectOptions = {}): ChartJsRangeSelect {
    let enabled = false;
    let startValue = 0;
    let endValue = 0;
    let chart: Chart | null = null;

    let isDragging = false;

    const setEnabled = (newEnabled: boolean) => {
        if (isDragging) {
            return;
        }

        enabled = newEnabled;
        if (chart) {
            chart.update();
        }
    }

    const setRange = (newStartValue: number, newEndValue: number) => {
        startValue = newStartValue;
        endValue = newEndValue;
        if (chart) {
            chart.update();
        }
    }

    const mouseDown = (event: MouseEvent) => {
        if (chart === null) {
            return;
        }

        isDragging = true;
        enabled = true;
        startValue = chart.scales["x"].getValueForPixel(event.offsetX) ?? 0;
        endValue = startValue;
        options.onRangeChange?.(startValue, endValue);
    }

    const mouseUp = () => {
        isDragging = false;
    }

    const mouseMove = (event: MouseEvent) => {
        if (chart === null || !isDragging) {
            return;
        }

        const newEndValue = chart.scales["x"].getValueForPixel(event.offsetX) ?? 0;
        setRange(startValue, newEndValue);
        options.onRangeChange?.(startValue, newEndValue);
    }

    const plugin: Plugin = {
        id: "range-select",

        start: (newChart) => {
            chart = newChart;
            chart.canvas.addEventListener("mousedown", mouseDown);
            chart.canvas.addEventListener("mouseup", mouseUp);
            chart.canvas.addEventListener("mousemove", mouseMove);
        },

        stop: (chart) => {
            chart.canvas?.removeEventListener("mousedown", mouseDown);
            chart.canvas?.removeEventListener("mouseup", mouseUp);
            chart.canvas?.removeEventListener("mousemove", mouseMove);
        },

        beforeDraw: (chart) => {
            if (!enabled) {
                return;
            }

            const ctx = chart.ctx;
            ctx.fillStyle = "rgba(0, 0, 0, 0.2)";
            const startX = chart.scales["x"].getPixelForValue(startValue);
            const endX = chart.scales["x"].getPixelForValue(endValue);
            ctx.fillRect(startX, chart.chartArea.top, endX - startX, chart.chartArea.height);
        }
    }

    return {
        setEnabled,
        setRange,
        plugin,
    }
}
