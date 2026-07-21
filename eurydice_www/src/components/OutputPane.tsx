import React from "react";
import { Bar, Line } from "react-chartjs-2";
import { Distribution, TupleDistribution, TupleFieldSchema } from "../util";
import {
  fieldName,
  computeMarginals,
  computeTuplePivot,
  computeTupleRows,
  TupleSort,
} from "../utils/tupleData";
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
  prepareCategoricalChartData,
  partitionDistributions,
  ColorGenerator,
  EnumDistributionGroup,
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
  const [showExportModal, setShowExportModal] = React.useState(false);
  const { sections } = React.useMemo(
    () => partitionDistributions(props.distributions),
    [props.distributions]
  );
  const tupleDistributions = props.tupleDistributions ?? [];

  const exportButton = (
    <button
      onClick={() => setShowExportModal(true)}
      className="btn btn-secondary"
    >
      Export
    </button>
  );

  return (
    <>
      <div className="flex flex-col gap-6">
        {sections.map((section, index) =>
          section.kind === "numeric" ? (
            <NumericOutputSection
              key="numeric"
              distributions={section.distributions}
              actions={index === 0 ? exportButton : undefined}
            />
          ) : (
            <EnumOutputSection
              key={`enum:${section.group.enumName}`}
              group={section.group}
              actions={index === 0 ? exportButton : undefined}
            />
          )
        )}
        {tupleDistributions.map(([name, distribution], index) => (
          <TupleOutputSection
            key={`tuple:${index}:${name}`}
            name={name}
            distribution={distribution}
          />
        ))}
      </div>

      <ExportModal
        distributions={props.distributions.map(([name, distribution]) => ({
          name,
          distribution,
        }))}
        isOpen={showExportModal}
        onClose={() => setShowExportModal(false)}
      />
    </>
  );
}

function NumericOutputSection({
  distributions,
  actions,
}: {
  distributions: [string, Distribution][];
  actions?: React.ReactNode;
}) {
  const [displayMode, setDisplayMode] = React.useState(
    DisplayMode.Distribution
  );
  const [tableMode, setTableMode] = React.useState(false);
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

  // Keep the range selection plugin's offset in sync with the minimum numeric outcome.
  React.useEffect(() => {
    const outcomes = distributions.flatMap(([, distribution]) =>
      distribution.probabilities.map(([outcome]) => outcome)
    );
    if (outcomes.length > 0) {
      plugin.current.setOffset(Math.min(...outcomes));
    }
  }, [distributions]);

  // Keep plugin enabled state in sync with the display mode.
  React.useEffect(() => {
    plugin.current.setEnabled(displayMode !== DisplayMode.Transposed);
  }, [displayMode]);

  const isDarkMode = React.useContext(DarkModeContext);
  const bracketUnavailableMessage =
    displayMode === DisplayMode.Transposed
      ? "Not available in transposed mode"
      : undefined;

  const setNumericDisplayMode = (mode: DisplayMode) => {
    setDisplayMode(mode);
    if (mode === DisplayMode.Transposed && showBracketing) {
      setShowBracketing(false);
      plugin.current.setActive(false);
    }
  };

  return (
    <section aria-label="Numeric outcomes">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span id="bracket-transposed-unavailable" className="sr-only">
          Not available in transposed mode
        </span>
        <div className="segmented" role="group" aria-label="Display mode">
          <button
            aria-pressed={displayMode === DisplayMode.Distribution}
            onClick={() => setDisplayMode(DisplayMode.Distribution)}
          >
            Distribution
          </button>
          <button
            aria-pressed={displayMode === DisplayMode.AtLeast}
            onClick={() => setNumericDisplayMode(DisplayMode.AtLeast)}
          >
            At least
          </button>
          <button
            aria-pressed={displayMode === DisplayMode.AtMost}
            onClick={() => setNumericDisplayMode(DisplayMode.AtMost)}
          >
            At most
          </button>
          <button
            aria-pressed={displayMode === DisplayMode.Transposed}
            onClick={() => setNumericDisplayMode(DisplayMode.Transposed)}
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
        <div className="relative">
          <button
            onClick={() => {
              if (displayMode === DisplayMode.Transposed) return;
              setShowBracketing(!showBracketing);
              plugin.current.setActive(!showBracketing);
              plugin.current.setRange(lowerBound, upperBound);
            }}
            aria-disabled={bracketUnavailableMessage !== undefined}
            aria-describedby={
              displayMode === DisplayMode.Transposed
                ? "bracket-transposed-unavailable"
                : undefined
            }
            data-tooltip={bracketUnavailableMessage}
            className={`btn-toggle${bracketUnavailableMessage ? " tooltip-control" : ""}`}
            aria-pressed={showBracketing && displayMode !== DisplayMode.Transposed}
          >
            Bracket {showBracketing ? "▲" : "▼"}
          </button>
        </div>
        {actions && <div className="ml-auto">{actions}</div>}
      </div>
      <div>
        {showBracketing && displayMode !== DisplayMode.Transposed && (
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
                    plugin.current.setRange(
                      newLowerBound,
                      Number(e.target.value)
                    );
                  }}
                  className="field"
                  style={{ width: "5em" }}
                />
              </label>
            </div>
            <BracketingTable
              distributions={distributions}
              lowerBound={lowerBound}
              upperBound={upperBound}
            />
          </div>
        )}
      </div>
      {tableMode ? (
        <CombinedProbabilityTable
          distributions={distributions}
          mode={displayMode}
        />
      ) : (
        <NumericChart
          distributions={distributions}
          mode={displayMode}
          isDarkMode={isDarkMode}
          plugin={plugin.current}
        />
      )}
    </section>
  );
}

function EnumOutputSection({
  group,
  actions,
}: {
  group: EnumDistributionGroup;
  actions?: React.ReactNode;
}) {
  const [tableMode, setTableMode] = React.useState(false);
  const isDarkMode = React.useContext(DarkModeContext);

  return (
    <section>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <h2 className="mr-auto text-sm font-semibold text-[var(--text-muted)]">
          {group.enumName}
        </h2>
        <button
          className="btn-toggle"
          aria-pressed={tableMode}
          onClick={() => setTableMode(!tableMode)}
        >
          Table
        </button>
        {actions}
      </div>
      {tableMode ? (
        <CombinedProbabilityTable
          distributions={group.distributions}
          mode={DisplayMode.Distribution}
          outcomes={group.labels.map((_, index) => index)}
          showStatistics={false}
        />
      ) : (
        <CategoricalChart group={group} isDarkMode={isDarkMode} />
      )}
    </section>
  );
}

type TupleView = "heatmap" | "table" | "list" | "marginals";

function TupleOutputSection({
  name,
  distribution,
}: {
  name: string;
  distribution: TupleDistribution;
}) {
  const isDarkMode = React.useContext(DarkModeContext);
  const arity = distribution.fields.length;

  // A 2-D joint fits a grid (heatmap / contingency table); higher arities fall
  // back to listing outcomes. Marginals are available for any arity.
  const views: { id: TupleView; label: string }[] =
    arity === 2
      ? [
          { id: "heatmap", label: "Heatmap" },
          { id: "table", label: "Table" },
          { id: "marginals", label: "Marginals" },
        ]
      : [
          { id: "list", label: "Table" },
          { id: "marginals", label: "Marginals" },
        ];
  const [view, setView] = React.useState<TupleView>(views[0].id);

  return (
    <section aria-label={`Tuple outcomes: ${name}`}>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <h2 className="mr-auto text-sm font-semibold text-[var(--text-muted)]">
          {name}
        </h2>
        <div className="segmented" role="group" aria-label="Tuple display mode">
          {views.map((v) => (
            <button
              key={v.id}
              aria-pressed={view === v.id}
              onClick={() => setView(v.id)}
            >
              {v.label}
            </button>
          ))}
        </div>
      </div>
      {(view === "heatmap" || view === "table") && (
        <TupleGrid
          distribution={distribution}
          variant={view === "heatmap" ? "heatmap" : "table"}
          isDarkMode={isDarkMode}
        />
      )}
      {view === "list" && <TupleListTable distribution={distribution} />}
      {view === "marginals" && (
        <TupleMarginals
          name={name}
          distribution={distribution}
          isDarkMode={isDarkMode}
        />
      )}
    </section>
  );
}

/** Largest number of cells we're willing to lay out for the 2-D grid views. */
const MAX_GRID_CELLS = 2500;

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}

function TupleGrid({
  distribution,
  variant,
  isDarkMode,
}: {
  distribution: TupleDistribution;
  variant: "heatmap" | "table";
  isDarkMode: boolean;
}) {
  const pivot = React.useMemo(
    () => computeTuplePivot(distribution),
    [distribution]
  );
  const { xAxis, yAxis, maxCell } = pivot;
  const [xField, yField] = distribution.fields;
  const cellCount = xAxis.values.length * yAxis.values.length;

  if (cellCount > MAX_GRID_CELLS) {
    return (
      <p className="rounded-lg border p-4 text-sm text-[var(--text-muted)]">
        This joint distribution has {xAxis.values.length}×{yAxis.values.length}{" "}
        cells — too many to lay out as a grid. Switch to the Marginals view.
      </p>
    );
  }

  const heatmap = variant === "heatmap";
  const accent = hexToRgb(isDarkMode ? "#3987e5" : "#2a78d6");
  const showCellText = !heatmap || cellCount <= 256;

  const cellBase = "px-2.5 py-1.5 text-right tabular-nums whitespace-nowrap";
  const colHeader =
    "px-2.5 py-1.5 text-right font-semibold whitespace-nowrap bg-[var(--surface-2)]";
  const rowHeader =
    "px-2.5 py-1.5 text-left font-semibold whitespace-nowrap bg-[var(--surface)]";
  const marginalCell = `${cellBase} bg-[var(--surface-2)] text-[var(--text-muted)]`;

  const pct = (p: number) => (p * 100).toFixed(2);

  return (
    <div className="dice-table overflow-x-auto rounded-lg border">
      <div className="px-3 py-2 text-xs text-[var(--text-muted)]">
        Columns: {fieldName(xField, 0)} · Rows: {fieldName(yField, 1)}
      </div>
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr>
            <th className={rowHeader} />
            {xAxis.values.map((x, i) => (
              <th key={x} className={colHeader}>
                {xAxis.labels[i]}
              </th>
            ))}
            <th className={colHeader}>Σ</th>
          </tr>
        </thead>
        <tbody>
          {yAxis.values.map((y, yi) => (
            <tr key={y} className="dice-row">
              <td className={rowHeader}>{yAxis.labels[yi]}</td>
              {xAxis.values.map((x, xi) => {
                const p = pivot.cell(x, y);
                const intensity = maxCell > 0 ? p / maxCell : 0;
                const style = heatmap
                  ? {
                      backgroundColor: `rgba(${accent[0]}, ${accent[1]}, ${accent[2]}, ${(
                        intensity * 0.9
                      ).toFixed(3)})`,
                      color: intensity > 0.5 ? "#ffffff" : undefined,
                    }
                  : undefined;
                return (
                  <td
                    key={x}
                    className={cellBase}
                    style={style}
                    title={`${xAxis.labels[xi]}, ${yAxis.labels[yi]}: ${pct(p)}%`}
                  >
                    {showCellText && p > 0
                      ? heatmap
                        ? (p * 100).toFixed(1)
                        : `${pct(p)}%`
                      : ""}
                  </td>
                );
              })}
              <td className={marginalCell}>{pct(pivot.yMarginal(y))}%</td>
            </tr>
          ))}
        </tbody>
        <tbody>
          <tr className="dice-row">
            <td className={`${rowHeader} text-[var(--text-muted)]`}>Σ</td>
            {xAxis.values.map((x) => (
              <td key={x} className={marginalCell}>
                {pct(pivot.xMarginal(x))}%
              </td>
            ))}
            <td className={marginalCell}>100.00%</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

/** Largest number of rows the list-out table renders before truncating. */
const MAX_LIST_ROWS = 1000;

function TupleListTable({ distribution }: { distribution: TupleDistribution }) {
  const [sort, setSort] = React.useState<TupleSort>("probability");
  const rows = React.useMemo(
    () => computeTupleRows(distribution, sort),
    [distribution, sort]
  );
  const shown = rows.slice(0, MAX_LIST_ROWS);
  const truncated = rows.length - shown.length;

  const cell = "px-3 py-1.5 text-right tabular-nums whitespace-nowrap";
  const fieldCell = "px-3 py-1.5 text-left whitespace-nowrap";
  const colHeader =
    "px-3 py-1.5 text-left font-semibold whitespace-nowrap sticky top-0 z-10 bg-[var(--surface-2)]";
  const probHeader =
    "px-3 py-1.5 text-right font-semibold whitespace-nowrap sticky top-0 z-10 bg-[var(--surface-2)]";

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="text-sm text-[var(--text-muted)]">Sort by</span>
        <div className="segmented" role="group" aria-label="Sort order">
          <button
            aria-pressed={sort === "probability"}
            onClick={() => setSort("probability")}
          >
            Probability
          </button>
          <button
            aria-pressed={sort === "lexicographic"}
            onClick={() => setSort("lexicographic")}
          >
            Outcome
          </button>
        </div>
      </div>
      <div className="dice-table overflow-x-auto rounded-lg border">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr>
              {distribution.fields.map((schema, i) => (
                <th key={i} className={colHeader}>
                  {fieldName(schema, i)}
                </th>
              ))}
              <th className={probHeader}>Probability</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((row, index) => (
              <tr key={index} className="dice-row">
                {row.labels.map((label, i) => (
                  <td key={i} className={fieldCell}>
                    {label}
                  </td>
                ))}
                <td className={cell}>{(row.probability * 100).toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {truncated > 0 && (
        <p className="mt-2 text-xs text-[var(--text-muted)]">
          Showing the {MAX_LIST_ROWS} most probable of {rows.length} outcomes.
        </p>
      )}
    </div>
  );
}

function TupleMarginals({
  name,
  distribution,
  isDarkMode,
}: {
  name: string;
  distribution: TupleDistribution;
  isDarkMode: boolean;
}) {
  const marginals = React.useMemo(
    () => computeMarginals(distribution),
    [distribution]
  );

  return (
    <div className="flex flex-col gap-6">
      {distribution.fields.map((schema: TupleFieldSchema, i) => (
        <div key={i}>
          <h3 className="mb-2 text-sm font-semibold text-[var(--text-muted)]">
            {fieldName(schema, i)}
          </h3>
          {schema.kind === "enum" ? (
            <CategoricalChart
              group={{
                enumName: schema.enumName,
                labels: schema.labels,
                distributions: [[name, marginals[i]]],
              }}
              isDarkMode={isDarkMode}
            />
          ) : (
            <NumericChart
              distributions={[[name, marginals[i]]]}
              mode={DisplayMode.Distribution}
              isDarkMode={isDarkMode}
            />
          )}
        </div>
      ))}
    </div>
  );
}

interface NumericChartProps {
  distributions: [string, Distribution][];
  mode: DisplayMode;
  isDarkMode: boolean;
  plugin?: ChartJsRangeSelect;
}

function NumericChart({ distributions, mode, isDarkMode, plugin }: NumericChartProps) {
  const gridColor = isDarkMode ? "#26323f" : "#e4eaf1";
  const textColor = isDarkMode ? "#94a3b8" : "#64748b";
  const tooltipBg = isDarkMode ? "#182230" : "#ffffff";
  const tooltipText = isDarkMode ? "#e5edf6" : "#1a2431";
  const tooltipBorder = isDarkMode ? "#3a4c60" : "#c2ccda";
  const grid = { color: gridColor, tickColor: gridColor };
  const datasets = prepareChartData(distributions, mode, isDarkMode);

  return (
    <div className="chart-container">
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
                label: (context) =>
                  `${context.dataset.label}: ${context.parsed.y.toFixed(2)}%`,
              },
            },
          },
        }}
        plugins={plugin ? [plugin.plugin] : []}
        width="100%"
        height="100%"
        style={{ userSelect: "none" }}
      />
    </div>
  );
}

function CategoricalChart({
  group,
  isDarkMode,
}: {
  group: EnumDistributionGroup;
  isDarkMode: boolean;
}) {
  const gridColor = isDarkMode ? "#26323f" : "#e4eaf1";
  const textColor = isDarkMode ? "#94a3b8" : "#64748b";
  const tooltipBg = isDarkMode ? "#182230" : "#ffffff";
  const tooltipText = isDarkMode ? "#e5edf6" : "#1a2431";
  const tooltipBorder = isDarkMode ? "#3a4c60" : "#c2ccda";
  const height = Math.max(180, group.labels.length * 42 + 70);

  return (
    <div className="relative" style={{ height }}>
      <Bar
        data={prepareCategoricalChartData(group, isDarkMode)}
        options={{
          indexAxis: "y",
          maintainAspectRatio: false,
          interaction: { intersect: false, mode: "nearest", axis: "y" },
          scales: {
            x: {
              beginAtZero: true,
              max: 100,
              border: { color: gridColor },
              ticks: {
                callback: (value) => `${value}%`,
                color: textColor,
                font: { size: 11 },
              },
              grid: { color: gridColor, tickColor: gridColor },
            },
            y: {
              border: { color: gridColor },
              ticks: { color: textColor, font: { size: 11 } },
              grid: { display: false },
            },
          },
          animation: false,
          plugins: {
            legend: {
              labels: {
                color: tooltipText,
                usePointStyle: true,
                pointStyle: "rect",
                boxWidth: 12,
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
                label: (context) =>
                  `${context.dataset.label}: ${context.parsed.x.toFixed(2)}%`,
              },
            },
          },
        }}
      />
    </div>
  );
}

export interface OutputPaneProps {
  distributions: [string, Distribution][];
  tupleDistributions?: [string, TupleDistribution][];
}

interface CombinedProbabilityTableProps {
  distributions: [string, Distribution][];
  mode: DisplayMode;
  outcomes?: number[];
  showStatistics?: boolean;
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
  outcomes,
  showStatistics = true,
}: CombinedProbabilityTableProps) {
  const isDarkMode = React.useContext(DarkModeContext);
  const colorGenerator = new ColorGenerator(isDarkMode);
  const colors = distributions.map(() => colorGenerator.nextColor());

  const sortedOutcomes = outcomes ?? getAllUniqueOutcomes(distributions);
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
        {showStatistics && <tbody className="dice-stats">
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
