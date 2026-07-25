import { NamedScalarDistribution, ScalarDistribution } from "../util";
import { DisplayMode } from "./chartData";

export interface TableRowData {
  outcome: number;
  outcomeLabel: string;
  values: string[];
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
  distributions: NamedScalarDistribution[]
): number[] {
  const allOutcomes = new Set<number>();
  distributions.forEach(([, distribution]) => {
    distribution.probabilities.forEach(([[outcome]]) => {
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
  sortedOutcomes: number[]
): TableRowData[] {
  return sortedOutcomes.map((outcome) => {
    const enumLabel = distributions
      .map(([, distribution]) => {
        const field = distribution.fields[0];
        return field.kind === "enum" ? field.labels[outcome] : undefined;
      })
      .find((label) => label !== undefined);
    const row = {
      outcome,
      outcomeLabel: enumLabel ?? outcome.toString(),
      values: [] as string[],
    };
    distributions.forEach(([, distribution]) => {
      const probability = Math.min(
        100,
        distribution.probabilities.reduce(
          (total, [[value], entryProbability]) => {
            const include =
              mode === DisplayMode.AtMost
                ? value <= outcome
                : mode === DisplayMode.AtLeast
                  ? value >= outcome
                  : value === outcome;
            return include ? total + entryProbability * 100 : total;
          },
          0
        )
      );

      row.values.push(probability > 0 ? `${probability.toFixed(2)}%` : "-");
    });
    return row;
  });
}

/**
 * Pre-computes statistics for each distribution
 */
export function computeDistributionStatistics(
  distributions: NamedScalarDistribution[]
): DistributionStatistics[] {
  return distributions.map(([, distribution]) => {
    const data = distribution.probabilities;
    const outcomes = data.map(([[outcome]]) => outcome);
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
}

/**
 * Calculates bracketing probabilities for a distribution within a given range
 */
export function calculateBracketingProbabilities(
  distribution: ScalarDistribution,
  lowerBound: number,
  upperBound: number
): BracketingProbabilities {
  let pLower = 0; // P(X < Lower)
  let pBetween = 0; // P(Lower <= X <= Upper)
  let pUpper = 0; // P(X > Upper)

  for (const [[outcome], probability] of distribution.probabilities) {
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
