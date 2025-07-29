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
        setLowerBound(startValue < endValue ? startValue : endValue);
        setUpperBound(startValue > endValue ? startValue : endValue);
      },
      onDragEnd: (startValue, endValue) => {
        if (startValue !== endValue) {
          // Avoid showing bracketing when the user has just made one click instead of a drag.
          setShowBracketing(true);
        }
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
    display = (
      <CombinedProbabilityTable
        distributions={props.distributions}
        mode={displayMode}
      />
    );
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
            mode: "index",
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
          plugins: {
            tooltip: {
              callbacks: {
                label: (context) => {
                  const value = context.parsed.y;
                  return `${context.dataset.label}: ${value.toFixed(3)}%`;
                },
              },
            },
          },
        }}
        plugins={[plugin.current.plugin]}
        width="100%"
        height="100%"
        // Prevent the canvas from being dragged when the user clicks and drags for bracketing.
        style={{userSelect: "none"}}
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
                    const newUpperBound = Math.max(
                      Number(e.target.value),
                      upperBound
                    );
                    setUpperBound(newUpperBound);
                    plugin.current.setRange(
                      Number(e.target.value),
                      newUpperBound
                    );
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
                    const newLowerBound = Math.min(
                      Number(e.target.value),
                      lowerBound
                    );
                    setLowerBound(newLowerBound);
                    plugin.current.setRange(newLowerBound, Number(e.target.value));
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
      <div className="relative" style={{aspectRatio: "1/1"}}>{display}</div>
    </>
  );
}

export interface OutputPaneProps {
  distributions: [string, Distribution][];
}

interface CombinedProbabilityTableProps {
  distributions: [string, Distribution][];
  mode: DisplayMode;
}

interface BracketingTableProps {
  distributions: [string, Distribution][];
  lowerBound: number;
  upperBound: number;
}

function CombinedProbabilityTable({
  distributions,
  mode,
}: CombinedProbabilityTableProps) {
  const colorGenerator = new ColorGenerator();
  const colors = distributions.map(() => colorGenerator.nextColor());

  // Get all unique outcomes across all distributions
  const allOutcomes = new Set<number>();
  distributions.forEach(([, distribution]) => {
    distribution.probabilities.forEach(([outcome]) => {
      allOutcomes.add(outcome);
    });
  });
  const sortedOutcomes = Array.from(allOutcomes).sort((a, b) => a - b);

  // Pre-compute all probability values for each outcome and distribution
  const tableData = sortedOutcomes.map((outcome) => {
    const row = { outcome, values: [] as string[] };
    distributions.forEach(([, distribution]) => {
      const probabilityEntry = distribution.probabilities.find(
        ([outcomeValue]) => outcomeValue === outcome
      );
      
      let probability = probabilityEntry ? probabilityEntry[1] * 100 : 0;
      
      // Apply mode transformations
      if (probability > 0) {
        const allProbs = distribution.probabilities.map(([, p]) => p * 100);
        const outcomes = distribution.probabilities.map(([o]) => o);
        const outcomeIndex = outcomes.indexOf(outcome);
        
        if (outcomeIndex !== -1) {
          switch (mode) {
            case DisplayMode.AtMost: {
              probability = partialSums(allProbs, false)[outcomeIndex];
              break;
            }
            case DisplayMode.AtLeast: {
              probability = partialSums(allProbs, true)[outcomeIndex];
              break;
            }
          }
        }
      }
      
      row.values.push(probability > 0 ? `${probability.toFixed(2)}%` : '-');
    });
    return row;
  });

  // Pre-compute statistics for each distribution
  const statisticsData = distributions.map(([, distribution]) => {
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

    return {
      mean: mean.toFixed(2),
      stdDev: stdDev.toFixed(2),
      min: min.toString(),
      max: max.toString(),
    };
  });

  const baseClassName = "border px-2 py-1 text-center";
  const headerClassName = "border px-2 py-1 text-center font-semibold sticky top-0 left-0 z-20";
  const leftColumnClassName = "border px-2 py-1 text-center font-semibold sticky left-0 z-10";

  return (
    <div className="overflow-x-auto">
      <table className="border-collapse border w-full">
        <thead>
          <tr>
            <th className={headerClassName}>Outcome</th>
            {distributions.map(([name], index) => (
              <th
                key={index}
                className="border px-2 py-1 text-center font-semibold"
                style={{ backgroundColor: colors[index] }}
              >
                {name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {tableData.map((row) => (
            <tr key={row.outcome}>
              <td className={leftColumnClassName}>{row.outcome}</td>
              {row.values.map((value, index) => (
                <td key={index} className={`${baseClassName} text-right`}>
                  {value}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      
      {/* Statistics table */}
      <div className="mt-4">
        <table className="border-collapse border w-full">
          <thead>
            <tr>
              <th className={headerClassName}>Statistic</th>
              {distributions.map(([name], index) => (
                <th
                  key={index}
                  className="border px-2 py-1 text-center font-semibold text-white"
                  style={{ backgroundColor: colors[index] }}
                >
                  {name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {['Mean', 'StdDev', 'Min', 'Max'].map((stat) => (
              <tr key={stat}>
                <td className={leftColumnClassName}>{stat}</td>
                {statisticsData.map((stats, index) => (
                  <td key={index} className={`${baseClassName} text-right`}>
                    {stats[stat.toLowerCase() as keyof typeof stats]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
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

