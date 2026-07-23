import { ChartData } from "chart.js";
import { ScalarDistribution } from "../util";

export type NamedDistribution = [string, ScalarDistribution];

export interface EnumDistributionGroup {
  enumName: string;
  labels: string[];
  distributions: NamedDistribution[];
}

export interface PartitionedDistributions {
  numeric: NamedDistribution[];
  enumGroups: EnumDistributionGroup[];
  sections: OutputSection[];
}

export type OutputSection =
  | { kind: "numeric"; distributions: NamedDistribution[] }
  | { kind: "enum"; group: EnumDistributionGroup };

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

export function partitionDistributions(
  distributions: NamedDistribution[]
): PartitionedDistributions {
  const numeric: NamedDistribution[] = [];
  const enumGroups = new Map<string, EnumDistributionGroup>();
  const sections: OutputSection[] = [];
  let hasNumericSection = false;

  for (const namedDistribution of distributions) {
    const [, distribution] = namedDistribution;
    if (distribution.enum_name === undefined) {
      if (!hasNumericSection) {
        sections.push({ kind: "numeric", distributions: numeric });
        hasNumericSection = true;
      }
      numeric.push(namedDistribution);
      continue;
    }

    let group = enumGroups.get(distribution.enum_name);
    if (group === undefined) {
      group = {
        enumName: distribution.enum_name,
        labels: distribution.labels ?? [],
        distributions: [],
      };
      enumGroups.set(distribution.enum_name, group);
      sections.push({ kind: "enum", group });
    }
    group.distributions.push(namedDistribution);
  }

  return {
    numeric,
    enumGroups: Array.from(enumGroups.values()),
    sections,
  };
}

export function numericOutcomeRange(
  distributions: NamedDistribution[]
): number | null {
  const numeric = distributions.filter(
    ([, distribution]) => distribution.enum_name === undefined
  );
  const outcomes = numeric.flatMap(([, distribution]) =>
    distribution.probabilities.map(([outcome]) => outcome)
  );
  if (outcomes.length === 0) return null;
  return Math.max(...outcomes) - Math.min(...outcomes);
}

export function prepareCategoricalChartData(
  group: EnumDistributionGroup,
  isDarkMode = false
): ChartData<"bar", number[], string> {
  const colorGenerator = new ColorGenerator(isDarkMode);
  return {
    labels: group.labels,
    datasets: group.distributions.map(([name, distribution]) => {
      const probabilities = new Map(distribution.probabilities);
      const color = colorGenerator.nextColor();
      return {
        label: name,
        data: group.labels.map((_, outcome) =>
          (probabilities.get(outcome) ?? 0) * 100
        ),
        backgroundColor: color,
        borderColor: color,
        borderWidth: 1,
      };
    }),
  };
}

export function prepareChartData(
  chartData: NamedDistribution[],
  mode: DisplayMode,
  isDarkMode = false
): ChartData<"line", number[], string> {
  if (mode === DisplayMode.Transposed) {
    return prepareTransposedChartData(chartData, isDarkMode);
  }

  // Compute the range of outcomes
  const outcomes = Array.from(chartData).flatMap((nameAndDist) => {
    return nameAndDist[1].probabilities.map(([x]) => x);
  });
  const min_outcome = Math.min(...outcomes);
  const max_outcome = Math.max(...outcomes);
  const range = Array.from(
    { length: max_outcome - min_outcome + 1 },
    (_, i) => i + min_outcome
  );
  const datasets = [];
  const colorGenerator = new ColorGenerator(isDarkMode);
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
    datasets.push({
      label: name,
      data,
      borderColor: color,
      backgroundColor: color,
    });
  }
  return {
    labels: range.map((x) => outcomeLabel(chartData, x)),
    datasets,
  };
}

function prepareTransposedChartData(
  chartData: NamedDistribution[],
  isDarkMode = false
): ChartData<"line", number[], string> {
  // Get all unique outcomes across all distributions
  const allOutcomes = new Set<number>();
  chartData.forEach(([, dist]) => {
    dist.probabilities.forEach(([outcome]) => {
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
      const outcomeProb = dist.probabilities.find(([val]) => val === outcome);
      const probability = outcomeProb ? outcomeProb[1] * 100 : 0;
      data.push(probability);
    }
    
    const color = colorGenerator.nextColor();
    datasets.push({
      label: outcomeLabel(chartData, outcome),
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

function outcomeLabel(chartData: NamedDistribution[], outcome: number): string {
  for (const [, distribution] of chartData) {
    const label = distribution.labels?.[outcome];
    if (label !== undefined) return label;
  }
  return outcome.toString();
}
