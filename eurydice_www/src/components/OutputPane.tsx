import React from "react";
import { Line } from "react-chartjs-2";
import { Distribution } from "../util";
import { Chart, ChartData, registerables } from "chart.js";
import { DarkModeContext } from "./DarkModeSwitcher";
Chart.register(...registerables);

export default function OutputPane(props: OutputPaneProps) {
  const [displayMode, setDisplayMode] = React.useState(
    DisplayMode.Distribution,
  );
  const [tableMode, setTableMode] = React.useState(false);

  const isDarkMode = React.useContext(DarkModeContext);
  const tickColor = isDarkMode ? "gray" : "lightgray";
  const gridColor = isDarkMode ? "gray" : "lightgray";
  const textColor = isDarkMode ? "white" : "lightgray";

  let display;
  if (tableMode) {
    const colorGenerator = new ColorGenerator();
    const colors = props.distributions.map(() => colorGenerator.nextColor());
    display = props.distributions.map(([name, dist], index) => (
      <ProbabilityTable
        key={index}
        name={name}
        distribution={dist}
        mode={displayMode}
        color={colors[index]}
      />
    ));
  } else {
    const datasets = prepareChartData(props.distributions, displayMode);
    const grid = {
      color: gridColor,
      tickColor,
    };
    display = (
      <Line
        data={datasets}
        options={{
          interaction: {
            intersect: false,
          },
          scales: {
            y: {
              beginAtZero: true,
              ticks: {
                callback: (value) => `${value}%`,
                color: textColor,
              },
              grid,
            },
            x: {
              ticks: {
                color: textColor,
              },
              grid,
            },
          },
          animation: false,
        }}
        width="100%"
        height="100%"
      />
    );
  }

  return (
    <>
      <div className="flex flex-row mb-4 px-2">
        <label className="border-2 border-blue-500 hover:border-blue-700 py-1 px-2 mr-1 rounded align-middle">
          <input
            type="checkbox"
            checked={tableMode}
            onChange={() => setTableMode(!tableMode)}
          />{" "}
          Table
        </label>
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
      {display}
    </>
  );
}

export interface OutputPaneProps {
  distributions: [string, Distribution][];
}

interface ProbabilityTableProps {
  name: string;
  distribution: Distribution;
  mode: DisplayMode;
  color: string;
}

function ProbabilityTable({
  name,
  distribution,
  mode,
  color,
}: ProbabilityTableProps) {
  const data = distribution.probabilities;
  const outcomes = data.map(([outcome]) => outcome);
  const probabilities = data.map(([, probability]) => probability);

  const mean = outcomes.reduce(
    (sum, val, i) => sum + val * probabilities[i],
    0,
  );
  const variance = outcomes.reduce(
    (sum, val, i) => sum + Math.pow(val - mean, 2) * probabilities[i],
    0,
  );
  const stdDev = Math.sqrt(variance);
  const min = Math.min(...outcomes);
  const max = Math.max(...outcomes);

  let probs = data.map(([_outcome, probability]) => probability * 100);
  switch (mode) {
    case DisplayMode.AtMost: {
      probs = partialSums(probs, false);
      break;
    }
    case DisplayMode.AtLeast: {
      probs = partialSums(probs, true);
      break;
    }
  }
  const outData = [];
  for (let i = outData.length; i < outcomes.length; i++) {
    outData[i] = [data[i][0], probs[i]];
  }

  const baseClassName = "border border-gray-300 px-1";

  return (
    <div className="inline-block m-2 align-top">
      <table className="border-collapse border border-gray-300">
        <thead>
          <tr>
            <th
              colSpan={2}
              className="text-white p-2 text-center"
              style={{ backgroundColor: color }}
            >
              {name}
            </th>
          </tr>
        </thead>
        <tbody className="text-sm">
          <tr>
            <td className={`${baseClassName} font-semibold`}>Mean</td>
            <td className={baseClassName}>{mean.toFixed(2)}</td>
          </tr>
          <tr>
            <td className={`${baseClassName} font-semibold`}>StdDev</td>
            <td className={baseClassName}>{stdDev.toFixed(2)}</td>
          </tr>
          <tr>
            <td className={`${baseClassName} font-semibold`}>Min</td>
            <td className={baseClassName}>{min}</td>
          </tr>
          <tr>
            <td className={`${baseClassName} font-semibold`}>Max</td>
            <td className={baseClassName}>{max}</td>
          </tr>
          {outData.map(([outcome, probability], index) => (
            <tr key={index}>
              <td className={baseClassName}>{outcome}</td>
              <td className={`${baseClassName} text-right`}>
                {probability.toFixed(2)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
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
  chartData: [string, Distribution][],
  mode: DisplayMode,
): ChartData<"line", number[], number> {
  // Compute the range of outcomes
  const outcomes = Array.from(chartData).flatMap((nameAndDist) => {
    return nameAndDist[1].probabilities.map(([x, _]) => x);
  });
  const min_outcome = Math.min(...outcomes);
  const max_outcome = Math.max(...outcomes);
  const range = Array.from(
    { length: max_outcome - min_outcome + 1 },
    (_, i) => i + min_outcome,
  );
  let datasets = [];
  const colorGenerator = new ColorGenerator();
  for (const nameAndDist of chartData) {
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

    const color = colorGenerator.nextColor();
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

class ColorGenerator {
  private rng: () => number;

  constructor() {
    this.rng = splitmix32(2);
  }

  nextColor(): string {
    return `rgba(${Math.floor(this.rng() * 256)}, ${Math.floor(this.rng() * 256)}, ${Math.floor(this.rng() * 256)}, 1.0)`;
  }
}
