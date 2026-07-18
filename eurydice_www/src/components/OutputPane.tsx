import React from "react";
import { Line } from "react-chartjs-2";
import { Distribution } from "../util";
import { Chart, registerables } from "chart.js";
import { DarkModeContext } from "./DarkModeSwitcher";
import ExportModal from "./ExportModal";
import {
  ChartJsRangeSelect,
  makeChartJsRangeSelect,
} from "../utils/chartJsRangeSelect";
import {
  DisplayMode,
  prepareChartData,
  ColorGenerator,
} from "../utils/chartData";
import {
  getAllUniqueOutcomes,
  computeTableData,
  computeDistributionStatistics,
  calculateBracketingProbabilities,
  DistributionStatistics,
} from "../utils/tableData";
Chart.register(...registerables);

export default function OutputPane(props: OutputPaneProps) {
  const [displayMode, setDisplayMode] = React.useState(
    DisplayMode.Distribution
  );
  const [tableMode, setTableMode] = React.useState(false);
  const [showBracketing, setShowBracketing] = React.useState(false);
  const [lowerBound, setLowerBound] = React.useState(0);
  const [upperBound, setUpperBound] = React.useState(0);
  const [showExportModal, setShowExportModal] = React.useState(false);
  const hasEnum = props.distributions.some(([, distribution]) => distribution.enum_name !== undefined);

  React.useEffect(() => {
    if (hasEnum) {
      setDisplayMode(DisplayMode.Distribution);
      setShowBracketing(false);
    }
  }, [hasEnum]);

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
  const gridColor = isDarkMode ? "#26323f" : "#e4eaf1";
  const textColor = isDarkMode ? "#94a3b8" : "#64748b";
  const tooltipBg = isDarkMode ? "#182230" : "#ffffff";
  const tooltipText = isDarkMode ? "#e5edf6" : "#1a2431";
  const tooltipBorder = isDarkMode ? "#3a4c60" : "#c2ccda";

  let display;
  if (tableMode) {
    display = (
      <CombinedProbabilityTable
        distributions={props.distributions}
        mode={displayMode}
      />
    );
  } else {
    const datasets = prepareChartData(props.distributions, displayMode, isDarkMode);
    const grid = {
      color: gridColor,
      tickColor: gridColor,
    };
    display = (
      <Line
        data={datasets}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          interaction: {
            intersect: false,
            mode: "index",
          },
          elements: {
            line: { borderWidth: 2 },
            point: { radius: 3, hoverRadius: 4 },
          },
          scales: {
            y: {
              beginAtZero: true,
              border: { color: gridColor },
              ticks: {
                callback: (value) => `${value}%`,
                color: textColor,
                font: { size: 11 },
              },
              grid,
            },
            x: {
              border: { color: gridColor },
              ticks: {
                color: textColor,
                font: { size: 11 },
                maxRotation: 0,
                autoSkipPadding: 12,
              },
              grid,
            },
          },
          animation: false,
          plugins: {
            legend: {
              labels: {
                color: tooltipText,
                usePointStyle: true,
                pointStyle: "rectRounded",
                boxWidth: 18,
                boxHeight: 12,
                padding: 16,
                font: { size: 12 },
              },
            },
            tooltip: {
              backgroundColor: tooltipBg,
              titleColor: tooltipText,
              bodyColor: tooltipText,
              borderColor: tooltipBorder,
              borderWidth: 1,
              padding: 10,
              cornerRadius: 8,
              usePointStyle: true,
              boxPadding: 4,
              callbacks: {
                label: (context) => {
                  const value = context.parsed.y;
                  return `${context.dataset.label}: ${value.toFixed(2)}%`;
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
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <div className="segmented" role="group" aria-label="Display mode">
          <button
            aria-pressed={displayMode === DisplayMode.Distribution}
            onClick={() => setDisplayMode(DisplayMode.Distribution)}
          >
            Distribution
          </button>
          <button
            disabled={hasEnum}
            aria-pressed={displayMode === DisplayMode.AtLeast}
            onClick={() => setDisplayMode(DisplayMode.AtLeast)}
          >
            At least
          </button>
          <button
            disabled={hasEnum}
            aria-pressed={displayMode === DisplayMode.AtMost}
            onClick={() => setDisplayMode(DisplayMode.AtMost)}
          >
            At most
          </button>
          <button
            aria-pressed={displayMode === DisplayMode.Transposed}
            onClick={() => {
              setDisplayMode(DisplayMode.Transposed);
              if (showBracketing) {
                setShowBracketing(false);
                plugin.current.setActive(false);
              }
            }}
          >
            Transposed
          </button>
        </div>
        <button
          className="btn-toggle"
          aria-pressed={tableMode}
          onClick={() => setTableMode(!tableMode)}
        >
          Table
        </button>
        <div className="relative ml-auto flex gap-2">
          <button
            onClick={() => {
              setShowBracketing(!showBracketing);
              plugin.current.setActive(!showBracketing);
              plugin.current.setRange(lowerBound, upperBound);
            }}
            disabled={hasEnum || displayMode === DisplayMode.Transposed}
            className="btn-toggle"
            aria-pressed={showBracketing && displayMode !== DisplayMode.Transposed}
          >
            Bracket {showBracketing ? "▲" : "▼"}
          </button>
          <button
            onClick={() => setShowExportModal(true)}
            className="btn btn-secondary"
          >
            Export
          </button>
        </div>
      </div>
      <div>
        {!hasEnum && showBracketing && displayMode !== DisplayMode.Transposed && (
          <div>
            <div className="mb-4 flex flex-wrap items-center gap-4">
              <label className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
                Lower
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
                  className="field"
                  style={{ width: "5em" }}
                />
              </label>
              <label className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
                Upper
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
                  className="field"
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
      <div className={tableMode ? "relative" : "chart-container"}>{display}</div>
      
      <ExportModal 
        distributions={props.distributions.map(([name, distribution]) => ({ name, distribution }))}
        isOpen={showExportModal}
        onClose={() => setShowExportModal(false)}
      />
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

function ColorSwatch({ color }: { color: string }) {
  return (
    <span
      className="mr-1.5 inline-block size-2.5 shrink-0 rounded-sm align-middle"
      style={{ backgroundColor: color }}
    />
  );
}

function CombinedProbabilityTable({
  distributions,
  mode,
}: CombinedProbabilityTableProps) {
  const isDarkMode = React.useContext(DarkModeContext);
  const colorGenerator = new ColorGenerator(isDarkMode);
  const colors = distributions.map(() => colorGenerator.nextColor());

  const sortedOutcomes = getAllUniqueOutcomes(distributions);
  const tableData = computeTableData(distributions, mode, sortedOutcomes);
  const statisticsData = computeDistributionStatistics(distributions);

  const cell = "px-3 py-1.5 text-right tabular-nums whitespace-nowrap";
  const cornerHeader =
    "px-3 py-1.5 text-left font-semibold sticky top-0 left-0 z-20 bg-[var(--surface-2)]";
  const colHeader =
    "px-3 py-1.5 text-right font-semibold whitespace-nowrap sticky top-0 z-10 bg-[var(--surface-2)]";
  const rowHeader =
    "px-3 py-1.5 text-left font-semibold sticky left-0 z-10 bg-[var(--surface)]";

  return (
    <div className="dice-table overflow-x-auto rounded-lg border">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr>
            <th className={cornerHeader}>Outcome</th>
            {distributions.map(([name], index) => (
              <th key={index} className={colHeader}>
                <ColorSwatch color={colors[index]} />
                {name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {tableData.map((row) => (
            <tr key={row.outcome} className="dice-row">
              <td className={rowHeader}>{row.outcomeLabel}</td>
              {row.values.map((value, index) => (
                <td key={index} className={cell}>
                  {value}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
        {!distributions.some(([, distribution]) => distribution.enum_name !== undefined) && <tbody className="dice-stats">
          {[
            ["Mean", "mean"],
            ["Std dev", "stdDev"],
            ["Min", "min"],
            ["Max", "max"],
          ].map(([stat, key]) => (
            <tr key={stat} className="dice-row">
              <td className={`${rowHeader} text-[var(--text-muted)]`}>{stat}</td>
              {statisticsData.map((stats, index) => (
                <td key={index} className={`${cell} text-[var(--text-muted)]`}>
                  {stats[key as keyof DistributionStatistics]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>}
      </table>
    </div>
  );
}

function BracketingTable({
  distributions,
  lowerBound,
  upperBound,
}: BracketingTableProps) {
  const isDarkMode = React.useContext(DarkModeContext);
  const colorGenerator = new ColorGenerator(isDarkMode);
  const colors = distributions.map(() => colorGenerator.nextColor());

  const cell = "px-3 py-1.5 text-right tabular-nums whitespace-nowrap";
  const header =
    "px-3 py-1.5 text-right font-semibold whitespace-nowrap bg-[var(--surface-2)]";
  const rowHeader = "px-3 py-1.5 text-left font-medium";

  return (
    <div className="dice-table mb-4 overflow-x-auto rounded-lg border">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr>
            <th className={`${header} text-left`}>Distribution</th>
            <th className={header}>P(X &lt; {lowerBound})</th>
            <th className={header}>
              P({lowerBound} ≤ X ≤ {upperBound})
            </th>
            <th className={header}>P(X &gt; {upperBound})</th>
          </tr>
        </thead>
        <tbody>
          {distributions.map(([name, distribution], index) => {
            const { pLower, pBetween, pUpper } = calculateBracketingProbabilities(
              distribution,
              lowerBound,
              upperBound
            );
            return (
              <tr key={index} className="dice-row">
                <td className={rowHeader}>
                  <ColorSwatch color={colors[index]} />
                  {name}
                </td>
                <td className={cell}>{pLower.toFixed(2)}%</td>
                <td className={cell}>{pBetween.toFixed(2)}%</td>
                <td className={cell}>{pUpper.toFixed(2)}%</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
