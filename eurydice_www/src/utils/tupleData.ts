import {
  Distribution,
  FieldSchema,
  outcomeValue,
  outcomeValues,
  ScalarDistribution,
} from "../util";

/** Display label for a single raw field value under its schema. */
export function fieldValueLabel(schema: FieldSchema, value: number): string {
  if (schema.kind === "categorical") {
    return schema.labels[value] ?? String(value);
  }
  return String(value);
}

/** A human-facing name for a field, used as a column/axis title. */
export function fieldName(dist: Distribution, index: number): string {
  return dist.fields[index].name ?? `Field ${index + 1}`;
}

export interface FieldAxis {
  /** Raw values along this axis, in display order. */
  values: number[];
  /** Display label for each value. */
  labels: string[];
}

/**
 * The ordered set of values a field ranges over. Categorical schemas contain
 * the values observed in the distribution; integer fields fill the observed
 * range so gaps render as empty cells, matching the 1-D numeric chart.
 */
export function fieldAxis(schema: FieldSchema, observed: number[]): FieldAxis {
  if (schema.kind === "categorical") {
    return {
      values: schema.labels.map((_, index) => index),
      labels: schema.labels.slice(),
    };
  }
  if (observed.length === 0) return { values: [], labels: [] };
  const min = Math.min(...observed);
  const max = Math.max(...observed);
  const values = Array.from({ length: max - min + 1 }, (_, i) => i + min);
  return { values, labels: values.map((value) => String(value)) };
}

export function observedValues(dist: Distribution, field: number): number[] {
  return dist.probabilities.map((_, outcome) =>
    outcomeValue(dist, outcome, field),
  );
}

/**
 * The marginal distribution of each field, obtained by summing joint
 * probabilities over all other fields. Categorical fields carry their labels through
 * so the result can feed the existing 1-D chart machinery.
 */
export function computeMarginals(dist: Distribution): ScalarDistribution[] {
  return dist.fields.map((field, index) => {
    const totals = new Map<number, number>();
    for (let outcome = 0; outcome < dist.probabilities.length; outcome++) {
      const value = outcomeValue(dist, outcome, index);
      const probability = dist.probabilities[outcome];
      totals.set(value, (totals.get(value) ?? 0) + probability);
    }
    const entries = Array.from(totals.entries()).sort((a, b) => a[0] - b[0]);
    return {
      fields: [field],
      values: entries.map(([value]) => value),
      probabilities: entries.map(([, probability]) => probability),
    };
  });
}

export interface TuplePivot {
  /** Field 0, laid out along the columns. */
  xAxis: FieldAxis;
  /** Field 1, laid out along the rows. */
  yAxis: FieldAxis;
  /** Joint probability of (field0 = x, field1 = y); 0 when absent. */
  cell: (x: number, y: number) => number;
  /** Marginal probability of field0 = x. */
  xMarginal: (x: number) => number;
  /** Marginal probability of field1 = y. */
  yMarginal: (y: number) => number;
  /** Largest joint probability, for scaling heatmap intensity. */
  maxCell: number;
}

/** Pivots an arity-2 joint distribution into a 2-D grid with marginals. */
export function computeTuplePivot(dist: Distribution): TuplePivot {
  const xAxis = fieldAxis(dist.fields[0].schema, observedValues(dist, 0));
  const yAxis = fieldAxis(dist.fields[1].schema, observedValues(dist, 1));
  const joint = new Map<string, number>();
  const xMarginal = new Map<number, number>();
  const yMarginal = new Map<number, number>();
  let maxCell = 0;
  for (let outcome = 0; outcome < dist.probabilities.length; outcome++) {
    const x = outcomeValue(dist, outcome, 0);
    const y = outcomeValue(dist, outcome, 1);
    const probability = dist.probabilities[outcome];
    const key = `${x},${y}`;
    const next = (joint.get(key) ?? 0) + probability;
    joint.set(key, next);
    if (next > maxCell) maxCell = next;
    xMarginal.set(x, (xMarginal.get(x) ?? 0) + probability);
    yMarginal.set(y, (yMarginal.get(y) ?? 0) + probability);
  }
  return {
    xAxis,
    yAxis,
    cell: (x, y) => joint.get(`${x},${y}`) ?? 0,
    xMarginal: (x) => xMarginal.get(x) ?? 0,
    yMarginal: (y) => yMarginal.get(y) ?? 0,
    maxCell,
  };
}

export type TupleSort = "probability" | "lexicographic";

export interface TupleRow {
  values: number[];
  labels: string[];
  probability: number;
}

/** Flattens a joint distribution into rows for the list-out table. */
export function computeTupleRows(
  dist: Distribution,
  sort: TupleSort,
): TupleRow[] {
  const rows: TupleRow[] = dist.probabilities.map((probability, outcome) => {
    const values = outcomeValues(dist, outcome);
    return {
      values,
      labels: values.map((value, field) =>
        fieldValueLabel(dist.fields[field].schema, value),
      ),
      probability,
    };
  });
  if (sort === "probability") {
    rows.sort((a, b) => b.probability - a.probability);
  } else {
    rows.sort((a, b) => {
      for (let i = 0; i < a.values.length; i++) {
        if (a.values[i] !== b.values[i]) return a.values[i] - b.values[i];
      }
      return 0;
    });
  }
  return rows;
}
