import { NamedScalarDistribution, ScalarDistribution } from "../util";
import {
  categoricalOutcomes,
  categoricalProbabilities,
  DisplayMode,
} from "./chartData";

export interface TableRowData {
  outcome: number | string;
  outcomeLabel: string;
  values: string[];
}

/** Table rows for a categorical chart containing numeric and symbol outputs. */
export function computeCategoricalTableData(
  distributions: NamedScalarDistribution[],
): TableRowData[] {
  const probabilities = distributions.map(([, distribution]) =>
    categoricalProbabilities(distribution),
  );
  return categoricalOutcomes(distributions).map(({ key, label }) => ({
    outcome: key,
    outcomeLabel: label,
    values: probabilities.map((distribution) => {
      const probability = (distribution.get(key) ?? 0) * 100;
      return probability > 0 ? `${probability.toFixed(2)}%` : "-";
    }),
  }));
}

export interface DistributionStatistics {
  mean: string;
  stdDev: string;
  min: string;
  max: string;
}

export interface BracketingProbabilities {
  pLower: number;
  pBetween: number;
  pUpper: number;
}

/**
 * Gets all unique outcomes across all distributions, sorted in ascending order
 */
export function getAllUniqueOutcomes(
  distributions: NamedScalarDistribution[],
): number[] {
  const allOutcomes = new Set<number>();
  distributions.forEach(([, distribution]) => {
    distribution.values.forEach((outcome) => {
      allOutcomes.add(outcome);
    });
  });
  return Array.from(allOutcomes).sort((a, b) => a - b);
}

/**
 * Pre-computes table data for the combined probability table
 */
export function computeTableData(
  distributions: NamedScalarDistribution[],
  mode: DisplayMode,
  sortedOutcomes: number[],
): TableRowData[] {
  return sortedOutcomes.map((outcome) => {
    const row = {
      outcome,
      outcomeLabel: outcome.toString(),
      values: [] as string[],
    };
    distributions.forEach(([, distribution]) => {
      let total = 0;
      for (let index = 0; index < distribution.probabilities.length; index++) {
        const value = distribution.values[index];
        const entryProbability = distribution.probabilities[index];
        const include =
          mode === DisplayMode.AtMost
            ? value <= outcome
            : mode === DisplayMode.AtLeast
              ? value >= outcome
              : value === outcome;
        if (include) total += entryProbability * 100;
      }
      const probability = Math.min(100, total);

      row.values.push(probability > 0 ? `${probability.toFixed(2)}%` : "-");
    });
    return row;
  });
}

/**
 * Pre-computes statistics for each distribution
 */
export function computeDistributionStatistics(
  distributions: NamedScalarDistribution[],
): DistributionStatistics[] {
  return distributions.map(([, distribution]) => {
    const outcomes = distribution.values;
    const probabilities = distribution.probabilities;

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

    return {
      mean: mean.toFixed(2),
      stdDev: stdDev.toFixed(2),
      min: min.toString(),
      max: max.toString(),
    };
  });
}

/**
 * Calculates bracketing probabilities for a distribution within a given range
 */
export function calculateBracketingProbabilities(
  distribution: ScalarDistribution,
  lowerBound: number,
  upperBound: number,
): BracketingProbabilities {
  let pLower = 0; // P(X < Lower)
  let pBetween = 0; // P(Lower <= X <= Upper)
  let pUpper = 0; // P(X > Upper)

  for (let index = 0; index < distribution.probabilities.length; index++) {
    const outcome = distribution.values[index];
    const probability = distribution.probabilities[index];
    if (outcome < lowerBound) {
      pLower += probability;
    } else if (outcome >= lowerBound && outcome <= upperBound) {
      pBetween += probability;
    } else if (outcome > upperBound) {
      pUpper += probability;
    }
  }

  return {
    pLower: pLower * 100,
    pBetween: pBetween * 100,
    pUpper: pUpper * 100,
  };
}
