import React from "react";
import { Bar, Line, Chart as ReactChart } from "react-chartjs-2";
import { MatrixController, MatrixElement } from "chartjs-chart-matrix";
import { Distribution, TupleDistribution } from "../util";
import {
  fieldName,
  computeMarginals,
  computeTuplePivot,
  computeTupleRows,
  TupleSort,
} from "../utils/tupleData";
import {
  Chart,
  registerables,
  ChartData,
  ChartOptions,
  ScriptableContext,
  TooltipItem,
} from "chart.js";
import { DarkModeContext } from "./DarkModeSwitcher";
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
Chart.register(...registerables, MatrixController, MatrixElement);

/** Shared light/dark palette for the Chart.js charts. */
function chartTheme(isDarkMode: boolean) {
  return {
    gridColor: isDarkMode ? "#26323f" : "#e4eaf1",
    textColor: isDarkMode ? "#94a3b8" : "#64748b",
    tooltipBg: isDarkMode ? "#182230" : "#ffffff",
    tooltipText: isDarkMode ? "#e5edf6" : "#1a2431",
    tooltipBorder: isDarkMode ? "#3a4c60" : "#c2ccda",
  };
}

/** Formats a probability in [0, 1] as a percentage string. */
function formatPercent(probability: number, digits = 2): string {
  return `${(probability * 100).toFixed(digits)}%`;
}

export default function OutputPane(props: OutputPaneProps) {
  const tupleDistributions = props.tupleDistributions ?? [];

  return (
    <div className="flex flex-col gap-6">
      <OutputSections distributions={props.distributions} />
      {tupleDistributions.map(([name, distribution], index) => (
        <TupleOutputSection
          key={`tuple:${index}:${name}`}
          name={name}
          distribution={distribution}
        />
      ))}
    </div>
  );
}

/**
 * Renders a set of named distributions as grouped numeric/enum sections. Shared
 * by the top-level output and by tuple marginals, so marginals get the same
 * full-featured numeric chart (display modes, bracketing, table view).
 */
function OutputSections({
  distributions,
}: {
  distributions: [string, Distribution][];
}) {
  const { sections } = React.useMemo(
    () => partitionDistributions(distributions),
    [distributions]
  );
  return (
    <>
      {sections.map((section) =>
        section.kind === "numeric" ? (
          <NumericOutputSection
            key="numeric"
            distributions={section.distributions}
          />
        ) : (
          <EnumOutputSection
            key={`enum:${section.group.enumName}`}
            group={section.group}
          />
        )
      )}
    </>
  );
}

function NumericOutputSection({
  distributions,
}: {
  distributions: [string, Distribution][];
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

function EnumOutputSection({ group }: { group: EnumDistributionGroup }) {
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

  // The selected view is remembered across edits, but editing the program can
  // change a tuple's arity and with it the available views (e.g. a 2-D heatmap
  // becoming a 3-D list). Fall back to the default whenever the remembered
  // choice is no longer offered, so the toggle and the body stay in sync.
  const activeView = views.some((v) => v.id === view) ? view : views[0].id;

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
              aria-pressed={activeView === v.id}
              onClick={() => setView(v.id)}
            >
              {v.label}
            </button>
          ))}
        </div>
      </div>
      {activeView === "heatmap" && (
        <TupleHeatmap distribution={distribution} isDarkMode={isDarkMode} />
      )}
      {activeView === "table" && (
        <TupleContingencyTable distribution={distribution} />
      )}
      {activeView === "list" && <TupleListTable distribution={distribution} />}
      {activeView === "marginals" && (
        <TupleMarginals distribution={distribution} />
      )}
    </section>
  );
}

/** Largest number of cells we're willing to lay out in the DOM contingency table. */
const MAX_TABLE_CELLS = 2500;

/** Beyond this many cells even a canvas heatmap stops being worthwhile. */
const MAX_HEATMAP_CELLS = 40000;

/**
 * Matt Zucker's polynomial approximation of the viridis colormap, for
 * `t` in [0, 1]. Viridis is perceptually uniform and colour-vision-deficiency
 * friendly, so it reads correctly in both light and dark themes. Returns an
 * `[r, g, b]` triple in 0–255.
 */
function viridis(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t));
  const c0 = [0.2777273272234177, 0.005407344544966578, 0.3340998053353061];
  const c1 = [0.1050930431085774, 1.404613529898575, 1.384590162594685];
  const c2 = [-0.3308618287255563, 0.214847559468213, 0.09509516302823659];
  const c3 = [-4.634230498983486, -5.799100973351585, -19.33244095627987];
  const c4 = [6.228269936347081, 14.17993336680509, 56.69055260068105];
  const c5 = [4.776384997670288, -13.74514537774601, -65.35303263337234];
  const c6 = [-5.435455855934631, 4.645852612178535, 26.3124352495832];
  return [0, 1, 2].map((i) => {
    const v =
      c0[i] +
      x * (c1[i] + x * (c2[i] + x * (c3[i] + x * (c4[i] + x * (c5[i] + x * c6[i])))));
    return Math.round(Math.min(1, Math.max(0, v)) * 255);
  }) as [number, number, number];
}

function viridisGradientCss(): string {
  const stops = [0, 0.25, 0.5, 0.75, 1].map((t) => {
    const [r, g, b] = viridis(t);
    return `rgb(${r}, ${g}, ${b}) ${Math.round(t * 100)}%`;
  });
  return `linear-gradient(to right, ${stops.join(", ")})`;
}

interface HeatmapCell {
  x: string;
  y: string;
  v: number;
}

/**
 * Canvas heatmap of a 2-D joint distribution. Unlike the contingency table it
 * reads as a graphic — coloured cells, no in-cell numbers, exact values on
 * hover — and it scales to grids far larger than the DOM table can handle.
 */
function TupleHeatmap({
  distribution,
  isDarkMode,
}: {
  distribution: TupleDistribution;
  isDarkMode: boolean;
}) {
  const pivot = React.useMemo(
    () => computeTuplePivot(distribution),
    [distribution]
  );
  const { xAxis, yAxis, maxCell } = pivot;
  const xCount = xAxis.values.length;
  const yCount = yAxis.values.length;

  const points = React.useMemo(() => {
    const out: HeatmapCell[] = [];
    xAxis.values.forEach((xv, xi) => {
      yAxis.values.forEach((yv, yi) => {
        const p = pivot.cell(xv, yv);
        // Only reachable outcomes get a cell; the rest stay as the background.
        if (p > 0) out.push({ x: xAxis.labels[xi], y: yAxis.labels[yi], v: p });
      });
    });
    return out;
  }, [pivot, xAxis, yAxis]);

  if (xCount * yCount > MAX_HEATMAP_CELLS) {
    return (
      <p className="rounded-lg border p-4 text-sm text-[var(--text-muted)]">
        This joint distribution has {xCount}×{yCount} cells — too many to draw.
        Switch to the Marginals view.
      </p>
    );
  }

  const { textColor, tooltipBg, tooltipText, tooltipBorder } =
    chartTheme(isDarkMode);

  const data: ChartData<"matrix", HeatmapCell[]> = {
    datasets: [
      {
        label: "joint",
        data: points,
        backgroundColor: (ctx: ScriptableContext<"matrix">) => {
          const value = (ctx.raw as HeatmapCell | undefined)?.v ?? 0;
          const [r, g, b] = viridis(maxCell > 0 ? value / maxCell : 0);
          return `rgb(${r}, ${g}, ${b})`;
        },
        borderWidth: 0,
        // Fill each category slot edge-to-edge. The extra pixel overlaps
        // neighbours just enough to hide antialiasing seams between cells.
        width: (ctx) => {
          const area = ctx.chart.chartArea;
          return area ? area.width / xCount + 1 : 0;
        },
        height: (ctx) => {
          const area = ctx.chart.chartArea;
          return area ? area.height / yCount + 1 : 0;
        },
      },
    ],
  };

  const options: ChartOptions<"matrix"> = {
    // Square-ish cells, clamped so a very lopsided grid can't grow absurdly tall.
    maintainAspectRatio: true,
    aspectRatio: Math.min(5, Math.max(0.6, xCount / yCount)),
    animation: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: tooltipBg,
        titleColor: tooltipText,
        bodyColor: tooltipText,
        borderColor: tooltipBorder,
        borderWidth: 1,
        padding: 10,
        cornerRadius: 8,
        displayColors: false,
        callbacks: {
          title: () => "",
          label: (ctx: TooltipItem<"matrix">) => {
            const point = ctx.raw as HeatmapCell;
            return `${fieldName(distribution, 0)}: ${point.x}, ${fieldName(distribution, 1)}: ${point.y} — ${formatPercent(point.v)}`;
          },
        },
      },
    },
    scales: {
      x: {
        type: "category",
        labels: xAxis.labels,
        offset: true,
        title: { display: true, text: fieldName(distribution, 0), color: textColor },
        ticks: { color: textColor, font: { size: 11 }, autoSkipPadding: 8 },
        grid: { display: false },
      },
      y: {
        type: "category",
        // Reverse so the first field value sits at the top, as in the table.
        labels: [...yAxis.labels].reverse(),
        offset: true,
        title: { display: true, text: fieldName(distribution, 1), color: textColor },
        ticks: { color: textColor, font: { size: 11 }, autoSkipPadding: 8 },
        grid: { display: false },
      },
    },
  };

  return (
    <div
      role="group"
      aria-label={`Joint distribution of ${fieldName(distribution, 0)} and ${fieldName(distribution, 1)}`}
    >
      <div className="mb-2 flex items-center gap-2 text-xs text-[var(--text-muted)]">
        <span>0%</span>
        <span
          className="h-2 w-24 rounded"
          style={{ background: viridisGradientCss() }}
        />
        <span>{formatPercent(maxCell)}</span>
      </div>
      {/* Keying on the grid shape remounts the chart when the axis cardinality
          changes, so `maintainAspectRatio` recomputes the canvas height instead
          of keeping the previous dimensions' size. */}
      <ReactChart
        key={`${xCount}x${yCount}`}
        type="matrix"
        data={data}
        options={options}
      />
    </div>
  );
}

/**
 * DOM contingency table for a 2-D joint distribution: exact per-cell
 * probabilities with row/column (Σ) marginals. Capped to a modest cell count;
 * larger joints belong in the heatmap or marginals views.
 */
function TupleContingencyTable({
  distribution,
}: {
  distribution: TupleDistribution;
}) {
  const pivot = React.useMemo(
    () => computeTuplePivot(distribution),
    [distribution]
  );
  const { xAxis, yAxis } = pivot;

  if (xAxis.values.length * yAxis.values.length > MAX_TABLE_CELLS) {
    return (
      <p className="rounded-lg border p-4 text-sm text-[var(--text-muted)]">
        This joint distribution has {xAxis.values.length}×{yAxis.values.length}{" "}
        cells — too many for a table. Switch to the Heatmap or Marginals view.
      </p>
    );
  }

  const cell = "px-2.5 py-1.5 text-right tabular-nums whitespace-nowrap";
  const colHeader =
    "px-2.5 py-1.5 text-right font-semibold whitespace-nowrap bg-[var(--surface-2)]";
  const rowHeader =
    "px-2.5 py-1.5 text-left font-semibold whitespace-nowrap bg-[var(--surface)]";
  const marginalCell = `${cell} bg-[var(--surface-2)] text-[var(--text-muted)]`;

  return (
    <div>
      <div className="mb-2 text-xs text-[var(--text-muted)]">
        Columns: {fieldName(distribution, 0)} · Rows: {fieldName(distribution, 1)}
      </div>
      <div className="dice-table overflow-x-auto rounded-lg border">
        <table
          className="w-full border-collapse text-sm"
          aria-label={`${fieldName(distribution, 1)} by ${fieldName(distribution, 0)}`}
        >
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
                  return (
                    <td
                      key={x}
                      className={cell}
                      title={`${fieldName(distribution, 0)}: ${xAxis.labels[xi]}, ${fieldName(distribution, 1)}: ${yAxis.labels[yi]} — ${formatPercent(p)}`}
                    >
                      {p > 0 ? formatPercent(p) : ""}
                    </td>
                  );
                })}
                <td className={marginalCell}>{formatPercent(pivot.yMarginal(y))}</td>
              </tr>
            ))}
          </tbody>
          <tbody>
            <tr className="dice-row">
              <td className={`${rowHeader} text-[var(--text-muted)]`}>Σ</td>
              {xAxis.values.map((x) => (
                <td key={x} className={marginalCell}>
                  {formatPercent(pivot.xMarginal(x))}
                </td>
              ))}
              <td className={marginalCell}>100.00%</td>
            </tr>
          </tbody>
        </table>
      </div>
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
        <table
          className="w-full border-collapse text-sm"
          aria-label={`Tuple outcomes: ${distribution.fields.map((_, i) => fieldName(distribution, i)).join(", ")}`}
        >
          <thead>
            <tr>
              {distribution.fields.map((_, i) => (
                <th key={i} className={colHeader}>
                  {fieldName(distribution, i)}
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
                <td className={cell}>{formatPercent(row.probability)}</td>
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

function TupleMarginals({ distribution }: { distribution: TupleDistribution }) {
  // Each field's marginal is a plain 1-D distribution, so route them through
  // the same section renderer as top-level outputs: numeric marginals overlay
  // on one full-featured chart (display modes, bracketing, table), and enum
  // marginals become categorical sections.
  const marginals = React.useMemo(
    () => computeMarginals(distribution),
    [distribution]
  );
  const named = React.useMemo(
    (): [string, Distribution][] =>
      marginals.map((marginal, i) => [fieldName(distribution, i), marginal]),
    [distribution, marginals]
  );

  return <OutputSections distributions={named} />;
}

interface NumericChartProps {
  distributions: [string, Distribution][];
  mode: DisplayMode;
  isDarkMode: boolean;
  plugin: ChartJsRangeSelect;
}

function NumericChart({ distributions, mode, isDarkMode, plugin }: NumericChartProps) {
  const { gridColor, textColor, tooltipBg, tooltipText, tooltipBorder } =
    chartTheme(isDarkMode);
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
        plugins={[plugin.plugin]}
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
  const { gridColor, textColor, tooltipBg, tooltipText, tooltipBorder } =
    chartTheme(isDarkMode);
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
