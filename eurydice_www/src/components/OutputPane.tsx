import React from "react";
import { Line } from "react-chartjs-2";
import { Distribution } from "../util";
import { Chart, ChartData, registerables } from "chart.js";
Chart.register(...registerables);

export default function OutputPane(props: OutputPaneProps) {
  const [displayMode, setDisplayMode] = React.useState(
    DisplayMode.Distribution
  );

  const datasets = prepareChartData(props.distributions, displayMode);
  return (
    <>
      <div className="flex flex-row mb-4 px-2">
        <label className="border-2 border-blue-500 hover:border-blue-700 py-1 px-2 mr-1 rounded align-middle">
          <input
            type="radio"
            name="displayMode"
            checked={displayMode === DisplayMode.Distribution}
            onChange={() => setDisplayMode(DisplayMode.Distribution)}
          />{" "}
          Distribution
        </label>
        <label className="border-2 border-blue-500 hover:border-blue-700 py-1 px-2 mr-1 rounded align-middle">
          <input
            type="radio"
            name="displayMode"
            checked={displayMode === DisplayMode.AtLeast}
            onChange={() => setDisplayMode(DisplayMode.AtLeast)}
          />{" "}
          At least
        </label>
        <label className="border-2 border-blue-500 hover:border-blue-700 py-1 px-2 mr-1 rounded align-middle">
          <input
            type="radio"
            name="displayMode"
            checked={displayMode === DisplayMode.AtMost}
            onChange={() => setDisplayMode(DisplayMode.AtMost)}
          />{" "}
          At most
        </label>
      </div>
      <Line
        data={datasets}
        options={{
          interaction: {
            // mode: "index",
            intersect: false,
          },
          scales: {
            y: {
              beginAtZero: true,
              ticks: {
                callback: (value) => `${value}%`,
              },
            },
          },
          animation: false,
        }}
        width="100%"
        height="100%"
      />
    </>
  );
}

export interface OutputPaneProps {
  distributions: Map<string, Distribution>;
}

enum DisplayMode {
  Distribution,
  AtMost,
  AtLeast,
}

/// A simple seedable random number generator
/// https://stackoverflow.com/a/47593316
function splitmix32(a: number) {
  return function () {
    a |= 0;
    a = (a + 0x9e3779b9) | 0;
    let t = a ^ (a >>> 16);
    t = Math.imul(t, 0x21f0aaad);
    t = t ^ (t >>> 15);
    t = Math.imul(t, 0x735a2d97);
    return ((t = t ^ (t >>> 15)) >>> 0) / 4294967296;
  };
}

function prepareChartData(
  chartData: Map<string, Distribution>,
  mode: DisplayMode
): ChartData<"line", number[], number> {
  // Compute the range of outcomes
  const outcomes = Array.from(chartData.values()).flatMap((dist) => {
    return dist.probabilities.map(([x, _]) => x);
  });
  const min_outcome = Math.min(...outcomes);
  const max_outcome = Math.max(...outcomes);
  const range = Array.from(
    { length: max_outcome - min_outcome + 1 },
    (_, i) => i + min_outcome
  );
  let datasets = [];
  const rng = splitmix32(2);
  for (const nameAndDist of chartData.entries()) {
    const [name, dist] = nameAndDist;
    const distMap = new Map(dist.probabilities);
    let data = range.map((x) => (distMap.get(x) ?? 0) * 100);

    switch (mode) {
      case DisplayMode.AtMost: {
        data = partialSums(data, false);
        break;
      }
      case DisplayMode.AtLeast: {
        data = partialSums(data, true);
        break;
      }
    }

    const color = `rgba(${Math.floor(rng() * 256)}, ${Math.floor(
      rng() * 256
    )}, ${Math.floor(rng() * 256)}, 1.0)`;
    datasets.push({ label: name, data, borderColor: color });
  }
  return {
    labels: range,
    datasets,
  };
}

function partialSums(array: number[], backwards: boolean): number[] {
  let sum = 0;
  const sums = [];
  for (let i = 0; i < array.length; i++) {
    if (backwards) {
      sum += array[array.length - 1 - i];
    } else {
      sum += array[i];
    }
    // Floating point errors can cause the sum to be slightly above 100, which screws
    // with the chart's scale.
    sums.push(Math.min(sum, 100));
  }
  if (backwards) {
    sums.reverse();
  }
  return sums;
}
