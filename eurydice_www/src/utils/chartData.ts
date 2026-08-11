import { ChartData } from "chart.js";
import {
  FieldSchema,
  NamedScalarDistribution,
  ScalarDistribution,
} from "../util";

export interface CategoricalOutcome {
  key: string;
  label: string;
}

export enum DisplayMode {
  Distribution,
  AtMost,
  AtLeast,
  Transposed,
}

export function partialSums(array: number[], backwards: boolean): number[] {
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

/**
 * Fixed categorical palette in a CVD-safe order (validated: worst adjacent
 * CVD ΔE 9.1 light / 8.4 dark). Assigned in order, never cycled by hue — a
 * ninth series simply wraps, which is acceptable here since series count is
 * user-controlled and rarely large. Light/dark columns are the same eight
 * hues stepped for their surface.
 */
const CATEGORICAL_LIGHT = [
  "#2a78d6", // blue
  "#008300", // green
  "#e87ba4", // magenta
  "#eda100", // yellow
  "#1baf7a", // aqua
  "#eb6834", // orange
  "#4a3aa7", // violet
  "#e34948", // red
];

const CATEGORICAL_DARK = [
  "#3987e5", // blue
  "#008300", // green
  "#d55181", // magenta
  "#c98500", // yellow
  "#199e70", // aqua
  "#d95926", // orange
  "#9085e9", // violet
  "#e66767", // red
];

export class ColorGenerator {
  private index = 0;
  private palette: string[];

  constructor(isDarkMode = false) {
    this.palette = isDarkMode ? CATEGORICAL_DARK : CATEGORICAL_LIGHT;
  }

  nextColor(): string {
    const color = this.palette[this.index % this.palette.length];
    this.index++;
    return color;
  }
}

function categoricalKey(schema: FieldSchema, outcome: number): string {
  return schema.kind === "int"
    ? `int:${outcome}`
    : `symbol:${schema.labels[outcome] ?? outcome}`;
}

/** Numeric outcomes first, then observed symbols in their first displayed order. */
export function categoricalOutcomes(
  distributions: NamedScalarDistribution[],
): CategoricalOutcome[] {
  const integers = new Set<number>();
  const symbols = new Map<string, string>();
  for (const [, distribution] of distributions) {
    const { schema } = distribution.fields[0];
    if (schema.kind === "int") {
      for (const outcome of distribution.values) {
        integers.add(outcome);
      }
    } else {
      // Symbol labels are a dense dictionary of the values observed in this
      // field, so there is no separate declared domain to filter here.
      for (const label of schema.labels) {
        symbols.set(`symbol:${label}`, label);
      }
    }
  }
  return [
    ...Array.from(integers)
      .sort((a, b) => a - b)
      .map((value) => ({ key: `int:${value}`, label: value.toString() })),
    ...Array.from(symbols, ([key, label]) => ({ key, label })),
  ];
}

export function categoricalProbabilities(
  distribution: ScalarDistribution,
): Map<string, number> {
  const { schema } = distribution.fields[0];
  const probabilities = new Map<string, number>();
  for (let index = 0; index < distribution.probabilities.length; index++) {
    const outcome = distribution.values[index];
    const probability = distribution.probabilities[index];
    const key = categoricalKey(schema, outcome);
    probabilities.set(key, (probabilities.get(key) ?? 0) + probability);
  }
  return probabilities;
}

/** Dense numeric range used by the line chart, or null for categorical data. */
export function numericChartOutcomeRange(
  distributions: NamedScalarDistribution[],
): number | null {
  if (
    distributions.some(
      ([, distribution]) =>
        distribution.fields[0].schema.kind === "categorical",
    )
  ) {
    return null;
  }
  const outcomes = distributions.flatMap(
    ([, distribution]) => distribution.values,
  );
  if (outcomes.length === 0) return null;
  return Math.max(...outcomes) - Math.min(...outcomes);
}

export function prepareCategoricalChartData(
  distributions: NamedScalarDistribution[],
  isDarkMode = false,
): ChartData<"bar", number[], string> {
  const outcomes = categoricalOutcomes(distributions);
  const colorGenerator = new ColorGenerator(isDarkMode);
  return {
    labels: outcomes.map(({ label }) => label),
    datasets: distributions.map(([name, distribution]) => {
      const probabilities = categoricalProbabilities(distribution);
      const color = colorGenerator.nextColor();
      return {
        label: name,
        data: outcomes.map(({ key }) => (probabilities.get(key) ?? 0) * 100),
        backgroundColor: color,
        borderColor: color,
        borderWidth: 1,
      };
    }),
  };
}

export function prepareChartData(
  chartData: NamedScalarDistribution[],
  mode: DisplayMode,
  isDarkMode = false,
): ChartData<"line", number[], string> {
  if (mode === DisplayMode.Transposed) {
    return prepareTransposedChartData(chartData, isDarkMode);
  }

  // Compute the range of outcomes
  const outcomes = Array.from(chartData).flatMap((nameAndDist) => {
    return nameAndDist[1].values;
  });
  const min_outcome = Math.min(...outcomes);
  const max_outcome = Math.max(...outcomes);
  const range = Array.from(
    { length: max_outcome - min_outcome + 1 },
    (_, i) => i + min_outcome,
  );
  const datasets = [];
  const colorGenerator = new ColorGenerator(isDarkMode);
  for (const nameAndDist of chartData) {
    const [name, dist] = nameAndDist;
    const distMap = new Map(
      dist.values.map((outcome, index) => [outcome, dist.probabilities[index]]),
    );
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
    datasets.push({
      label: name,
      data,
      borderColor: color,
      backgroundColor: color,
    });
  }
  return {
    labels: range.map(String),
    datasets,
  };
}

function prepareTransposedChartData(
  chartData: NamedScalarDistribution[],
  isDarkMode = false,
): ChartData<"line", number[], string> {
  // Get all unique outcomes across all distributions
  const allOutcomes = new Set<number>();
  chartData.forEach(([, dist]) => {
    dist.values.forEach((outcome) => {
      allOutcomes.add(outcome);
    });
  });

  const sortedOutcomes = Array.from(allOutcomes).sort((a, b) => a - b);
  const distributionNames = chartData.map(([name]) => name);

  // Create a dataset for each outcome value
  const datasets = [];
  const colorGenerator = new ColorGenerator(isDarkMode);

  for (const outcome of sortedOutcomes) {
    const data: number[] = [];

    // For each distribution, get the probability of this outcome
    for (const [, dist] of chartData) {
      const index = dist.values.findIndex((value) => value === outcome);
      const probability = index >= 0 ? dist.probabilities[index] * 100 : 0;
      data.push(probability);
    }

    const color = colorGenerator.nextColor();
    datasets.push({
      label: outcome.toString(),
      data,
      borderColor: color,
      backgroundColor: color,
    });
  }

  return {
    labels: distributionNames,
    datasets,
  };
}
