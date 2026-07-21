import React from "react";
import { Bar, Line } from "react-chartjs-2";
import { Distribution, TupleDistribution } from "../util";
import {
  fieldName,
  computeMarginals,
  computeTuplePivot,
  computeTupleRows,
  TupleSort,
} from "../utils/tupleData";
import { Chart, registerables } from "chart.js";
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
Chart.register(...registerables);

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
      {(activeView === "heatmap" || activeView === "table") && (
        <TupleGrid
          distribution={distribution}
          variant={activeView === "heatmap" ? "heatmap" : "table"}
        />
      )}
      {activeView === "list" && <TupleListTable distribution={distribution} />}
      {activeView === "marginals" && (
        <TupleMarginals distribution={distribution} />
      )}
    </section>
  );
}

/** Largest number of cells we're willing to lay out for the 2-D grid views. */
const MAX_GRID_CELLS = 2500;

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

/** Black or white text, whichever contrasts better with the given colour. */
function textOn([r, g, b]: [number, number, number]): string {
  return 0.299 * r + 0.587 * g + 0.114 * b > 140 ? "#1a2431" : "#ffffff";
}

function viridisGradientCss(): string {
  const stops = [0, 0.25, 0.5, 0.75, 1].map((t) => {
    const [r, g, b] = viridis(t);
    return `rgb(${r}, ${g}, ${b}) ${Math.round(t * 100)}%`;
  });
  return `linear-gradient(to right, ${stops.join(", ")})`;
}

function TupleGrid({
  distribution,
  variant,
}: {
  distribution: TupleDistribution;
  variant: "heatmap" | "table";
}) {
  const pivot = React.useMemo(
    () => computeTuplePivot(distribution),
    [distribution]
  );
  const { xAxis, yAxis, maxCell } = pivot;
  const [xField, yField] = distribution.fields;
  const cellCount = xAxis.values.length * yAxis.values.length;
  const heatmap = variant === "heatmap";

  // Whether a percentage fits in a heatmap cell depends on the pane width and
  // the column count, so measure the scroll container and hide the in-cell text
  // once columns get too narrow — the value stays available via the cell's
  // hover tooltip.
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const [textFits, setTextFits] = React.useState(true);
  React.useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const update = () => {
      const ROW_LABEL_WIDTH = 72;
      const perColumn =
        (el.clientWidth - ROW_LABEL_WIDTH) / (xAxis.values.length + 1);
      setTextFits(perColumn >= 46);
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, [xAxis.values.length]);

  if (cellCount > MAX_GRID_CELLS) {
    return (
      <p className="rounded-lg border p-4 text-sm text-[var(--text-muted)]">
        This joint distribution has {xAxis.values.length}×{yAxis.values.length}{" "}
        cells — too many to lay out as a grid. Switch to the Marginals view.
      </p>
    );
  }

  const showCellText = !heatmap || textFits;

  const cellBase = "px-2.5 py-1.5 tabular-nums whitespace-nowrap";
  const valueCell = `${cellBase} ${heatmap ? "text-center" : "text-right"}`;
  const colHeader =
    "px-2.5 py-1.5 text-right font-semibold whitespace-nowrap bg-[var(--surface-2)]";
  const rowHeader =
    "px-2.5 py-1.5 text-left font-semibold whitespace-nowrap bg-[var(--surface)]";
  const marginalCell = `${cellBase} text-right bg-[var(--surface-2)] text-[var(--text-muted)]`;

  const pct = (p: number) => (p * 100).toFixed(2);

  return (
    <div>
      {/* Caption and legend sit outside the horizontal scroll area so they stay
          in view while a wide grid scrolls. */}
      <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-[var(--text-muted)]">
        <span>
          Columns: {fieldName(xField, 0)} · Rows: {fieldName(yField, 1)}
        </span>
        {heatmap && (
          <span className="flex items-center gap-2">
            <span>0%</span>
            <span
              className="h-2 w-24 rounded"
              style={{ background: viridisGradientCss() }}
            />
            <span>{pct(maxCell)}%</span>
          </span>
        )}
      </div>
      <div
        ref={scrollRef}
        className="dice-table overflow-x-auto rounded-lg border"
      >
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
                // Zero-probability cells stay on the surface so the reachable
                // outcomes are the only ones that carry colour.
                let style: React.CSSProperties | undefined;
                if (heatmap && p > 0) {
                  const rgb = viridis(maxCell > 0 ? p / maxCell : 0);
                  style = {
                    backgroundColor: `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`,
                    color: textOn(rgb),
                  };
                }
                return (
                  <td
                    key={x}
                    className={valueCell}
                    style={style}
                    title={`${xAxis.labels[xi]}, ${yAxis.labels[yi]}: ${pct(p)}%`}
                  >
                    {showCellText && p > 0
                      ? heatmap
                        ? `${(p * 100).toFixed(1)}%`
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
      marginals.map((marginal, i) => [`Field ${i + 1}`, marginal]),
    [marginals]
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
