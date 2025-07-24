import { ChartData } from "chart.js";
import { Distribution } from "../util";

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

export class ColorGenerator {
  private rng: () => number;

  constructor() {
    this.rng = splitmix32(2);
  }

  nextColor(): string {
    return `rgba(${Math.floor(this.rng() * 256)}, ${Math.floor(
      this.rng() * 256
    )}, ${Math.floor(this.rng() * 256)}, 1.0)`;
  }
}

export function prepareChartData(
  chartData: [string, Distribution][],
  mode: DisplayMode
): ChartData<"line", number[], string> {
  if (mode === DisplayMode.Transposed) {
    return prepareTransposedChartData(chartData);
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
    labels: range.map((x) => x.toString()),
    datasets,
  };
}

function prepareTransposedChartData(
  chartData: [string, Distribution][]
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
  const colorGenerator = new ColorGenerator();
  
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
      label: `${outcome}`,
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