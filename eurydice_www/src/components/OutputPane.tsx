import React from "react";
import { Line } from "react-chartjs-2";
import { Distribution } from "../util";
import { Chart, registerables } from "chart.js";
import { DarkModeContext } from "./DarkModeSwitcher";
import {
  generateValuesOnlyCSV,
  generateAnyDiceFormatCSV,
  downloadCSV,
  DistributionData,
} from "../utils/csvExport";
import {
  ChartJsRangeSelect,
  makeChartJsRangeSelect,
} from "../utils/chartJsRangeSelect";
import {
  DisplayMode,
  prepareChartData,
  ColorGenerator,
  partialSums,
} from "../utils/chartData";
Chart.register(...registerables);

export default function OutputPane(props: OutputPaneProps) {
  const [displayMode, setDisplayMode] = React.useState(
    DisplayMode.Distribution
  );
  const [tableMode, setTableMode] = React.useState(false);
  const [showExportMenu, setShowExportMenu] = React.useState(false);
  const [showBracketing, setShowBracketing] = React.useState(false);
  const [lowerBound, setLowerBound] = React.useState(0);
  const [upperBound, setUpperBound] = React.useState(0);

  // The plugin is a ref because we can't recreate it every time the distributions change.
  const plugin = React.useRef<ChartJsRangeSelect>(
    makeChartJsRangeSelect({
      onRangeChange: (startValue, endValue) => {
        setShowBracketing(true);
        setLowerBound(startValue < endValue ? startValue : endValue);
        setUpperBound(startValue > endValue ? startValue : endValue);
      },
    })
  );

  // Keep the range selection plugin's offset in sync with the minimum outcome.
  React.useEffect(() => {
    plugin.current.setOffset(Math.min(
      ...Array.from(props.distributions)
        .map(([, distribution]) =>
          distribution.probabilities.map(([x]) => x)
        )
        .flat()
    ));
  }, [props.distributions]);

  // Keep plugin enabled state in sync with display mode
  React.useEffect(() => {
    plugin.current.setEnabled(displayMode !== DisplayMode.Transposed);
  }, [displayMode]);

  const isDarkMode = React.useContext(DarkModeContext);
  const tickColor = isDarkMode ? "gray" : "lightgray";
  const gridColor = isDarkMode ? "gray" : "lightgray";
  const textColor = isDarkMode ? "white" : "lightgray";

  const handleExport = (
    generate: (distributions: DistributionData[]) => string
  ) => {
    const distributionData = props.distributions.map(
      ([name, distribution]) => ({
        name,
        distribution,
      })
    );
    const csv = generate(distributionData);
    downloadCSV(csv, "distributions_values.csv");
    setShowExportMenu(false);
  };

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
        plugins={[plugin.current.plugin]}
        width="100%"
        height="100%"
      />
    );
  }

  return (
    <>
      <div className="flex flex-wrap items-center gap-1 mb-4 px-2">
        <label className="border-2 border-blue-500 hover:border-blue-700 py-1 px-2 rounded-sm align-middle">
          <input
            type="checkbox"
            checked={tableMode}
            onChange={() => setTableMode(!tableMode)}
          />{" "}
          Table
        </label>
        <label className="border-2 border-blue-500 hover:border-blue-700 py-1 px-2 rounded-sm align-middle">
          <input
            type="radio"
            name="displayMode"
            checked={displayMode === DisplayMode.Distribution}
            onChange={() => setDisplayMode(DisplayMode.Distribution)}
          />{" "}
          Distribution
        </label>
        <label className="border-2 border-blue-500 hover:border-blue-700 py-1 px-2 rounded-sm align-middle">
          <input
            type="radio"
            name="displayMode"
            checked={displayMode === DisplayMode.AtLeast}
            onChange={() => setDisplayMode(DisplayMode.AtLeast)}
          />{" "}
          At least
        </label>
        <label className="border-2 border-blue-500 hover:border-blue-700 py-1 px-2 rounded-sm align-middle">
          <input
            type="radio"
            name="displayMode"
            checked={displayMode === DisplayMode.AtMost}
            onChange={() => setDisplayMode(DisplayMode.AtMost)}
          />{" "}
          At most
        </label>
        <label className="border-2 border-blue-500 hover:border-blue-700 py-1 px-2 rounded-sm align-middle">
          <input
            type="radio"
            name="displayMode"
            checked={displayMode === DisplayMode.Transposed}
            onChange={() => {
              setDisplayMode(DisplayMode.Transposed);
              if (showBracketing) {
                setShowBracketing(false);
                plugin.current.setActive(false);
              }
            }}
          />{" "}
          Transposed
        </label>
        <div className="relative flex gap-1 ml-auto">
          <button
            onClick={() => {
              setShowBracketing(!showBracketing);
              plugin.current.setActive(!showBracketing);
              plugin.current.setRange(lowerBound, upperBound);
            }}
            disabled={displayMode === DisplayMode.Transposed}
            className={`border-2 py-1 px-3 rounded-sm ${
              displayMode === DisplayMode.Transposed
                ? "border-gray-400 bg-gray-400 text-gray-600 cursor-not-allowed"
                : "border-green-500 hover:border-green-700 bg-green-500 hover:bg-green-600"
            }`}
          >
            Bracket {showBracketing ? "▲" : "▼"}
          </button>
          <button
            onClick={() => setShowExportMenu(!showExportMenu)}
            className="border-2 border-green-500 hover:border-green-700 bg-green-500 hover:bg-green-600 py-1 px-3 rounded-sm"
          >
            Export {showExportMenu ? "▲" : "▼"}
          </button>
          {showExportMenu && (
            <div className="absolute right-0 top-full mt-1 bg-white border border-gray-300 rounded shadow-lg z-10 min-w-48">
              <button
                onClick={() => handleExport(generateValuesOnlyCSV)}
                className="block w-full text-left px-4 py-2 hover:bg-gray-100 text-black"
              >
                CSV (Values Only)
              </button>
              <button
                onClick={() => handleExport(generateAnyDiceFormatCSV)}
                className="block w-full text-left px-4 py-2 hover:bg-gray-100 text-black"
              >
                CSV (AnyDice Format)
              </button>
            </div>
          )}
        </div>
      </div>
      <div>
        {showBracketing && displayMode !== DisplayMode.Transposed && (
          <div>
            <div className="mb-4">
              <label className="mr-1">
                Lower:
                <input
                  type="number"
                  value={lowerBound}
                  onChange={(e) => {
                    setLowerBound(Number(e.target.value));
                    plugin.current.setRange(Number(e.target.value), upperBound);
                  }}
                  className="w-20 border rounded px-2 py-1 mx-2"
                  style={{ width: "5em" }}
                />
              </label>
              <label className="ml-2">
                Upper:
                <input
                  type="number"
                  value={upperBound}
                  onChange={(e) => {
                    setUpperBound(Number(e.target.value));
                    plugin.current.setRange(lowerBound, Number(e.target.value));
                  }}
                  className="w-20 border rounded px-2 py-1 mx-2"
                  style={{ width: "5em" }}
                />
              </label>
            </div>
            <BracketingTable
              distributions={props.distributions}
              lowerBound={lowerBound}
              upperBound={upperBound}
            />
          </div>
        )}
      </div>
      <div className="relative">{display}</div>
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

interface BracketingTableProps {
  distributions: [string, Distribution][];
  lowerBound: number;
  upperBound: number;
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
    0
  );
  const variance = outcomes.reduce(
    (sum, val, i) => sum + Math.pow(val - mean, 2) * probabilities[i],
    0
  );
  const stdDev = Math.sqrt(variance);
  const min = Math.min(...outcomes);
  const max = Math.max(...outcomes);

  let probs = data.map(([, probability]) => probability * 100);
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

function BracketingTable({
  distributions,
  lowerBound,
  upperBound,
}: BracketingTableProps) {
  const calculateProbabilities = (
    distribution: Distribution,
    lower: number,
    upper: number
  ) => {
    let pLower = 0; // P(X < Lower)
    let pBetween = 0; // P(Lower <= X <= Upper)
    let pUpper = 0; // P(X > Upper)

    for (const [outcome, probability] of distribution.probabilities) {
      if (outcome < lower) {
        pLower += probability;
      } else if (outcome >= lower && outcome <= upper) {
        pBetween += probability;
      } else if (outcome > upper) {
        pUpper += probability;
      }
    }

    return {
      pLower: pLower * 100,
      pBetween: pBetween * 100,
      pUpper: pUpper * 100,
    };
  };

  const baseClassName = "border border-gray-300 px-2 py-1 text-center";
  const headerClassName =
    "border border-gray-300 px-2 py-1 text-center font-semibold";
  const colorGenerator = new ColorGenerator();
  const colors = distributions.map(() => colorGenerator.nextColor());

  return (
    <div className="mb-4">
      <table className="border-collapse border border-gray-300 w-full">
        <thead>
          <tr>
            <th className={headerClassName}>Distribution</th>
            <th className={headerClassName}>P(X &lt; {lowerBound})</th>
            <th className={headerClassName}>
              P({lowerBound} ≤ X ≤ {upperBound})
            </th>
            <th className={headerClassName}>P(X &gt; {upperBound})</th>
          </tr>
        </thead>
        <tbody>
          {distributions.map(([name, distribution], index) => {
            const { pLower, pBetween, pUpper } = calculateProbabilities(
              distribution,
              lowerBound,
              upperBound
            );
            return (
              <tr key={index}>
                <td
                  className="border border-gray-300 px-2 py-1"
                  style={{ backgroundColor: colors[index] }}
                >
                  {name}
                </td>
                <td className={baseClassName}>{pLower.toFixed(2)}%</td>
                <td className={baseClassName}>{pBetween.toFixed(2)}%</td>
                <td className={baseClassName}>{pUpper.toFixed(2)}%</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

