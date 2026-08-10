import { describe, it, expect } from "vitest";
import { Distribution } from "../util";
import {
  normalizeFieldSchema,
  normalizeDistribution,
  fieldName,
  fieldValueLabel,
  fieldAxis,
  computeMarginals,
  computeTuplePivot,
  computeTupleRows,
} from "./tupleData";

// A 2x2 joint distribution: field 0 is an int (1..2), field 1 is an enum.
const jointIntEnum: Distribution = {
  fields: [
    { kind: "int" },
    { kind: "categorical", labels: ["MISS", "HIT"] },
  ],
  entries: [
    [[1, 0], 0.1],
    [[1, 1], 0.2],
    [[2, 0], 0.3],
    [[2, 1], 0.4],
  ],
};

describe("tupleData normalization", () => {
  it("normalizes wire field schemas", () => {
    expect(normalizeFieldSchema("Int")).toEqual({ kind: "int" });
    expect(
      normalizeFieldSchema({ Categorical: { labels: ["A", "B"] } })
    ).toEqual({ kind: "categorical", labels: ["A", "B"] });
  });

  it("passes distribution probabilities through", () => {
    const dist = normalizeDistribution({
      fields: ["Int", { Categorical: { labels: ["A"] } }],
      field_names: ["Count", "Result"],
      entries: [[[1, 0], 1]],
    });
    expect(dist.fields[1]).toEqual({ kind: "categorical", labels: ["A"] });
    expect(dist.fieldNames).toEqual(["Count", "Result"]);
  });

  it("uses the same wire shape for scalar distributions", () => {
    const dist = normalizeDistribution({
      fields: [{ Categorical: { labels: ["A", "B"] } }],
      entries: [[[1], 1]],
    });
    expect(dist.fields).toEqual([
      { kind: "categorical", labels: ["A", "B"] },
    ]);
    expect(dist.entries).toEqual([[[1], 1]]);
  });
});

describe("tupleData labels and axes", () => {
  it("prefers explicit field names and otherwise uses existing defaults", () => {
    expect(fieldName(jointIntEnum, 0)).toBe("Field 1");
    expect(fieldName(jointIntEnum, 1)).toBe("Field 2");
    expect(
      fieldName({ ...jointIntEnum, fieldNames: ["Roll", "Outcome"] }, 1)
    ).toBe("Outcome");
  });

  it("labels enum fields by member name and ints by value", () => {
    expect(fieldValueLabel({ kind: "int" }, 7)).toBe("7");
    expect(
      fieldValueLabel({ kind: "categorical", labels: ["MISS", "HIT"] }, 1)
    ).toBe("HIT");
  });

  it("fills the observed range for int axes", () => {
    expect(fieldAxis({ kind: "int" }, [2, 4]).values).toEqual([2, 3, 4]);
  });

  it("uses every observed symbol in the field dictionary", () => {
    const axis = fieldAxis(
      { kind: "categorical", labels: ["A", "C"] },
      [0, 1]
    );
    expect(axis.values).toEqual([0, 1]);
    expect(axis.labels).toEqual(["A", "C"]);
  });
});

describe("tupleData marginals", () => {
  it("sums out other fields", () => {
    const [marginal0, marginal1] = computeMarginals(jointIntEnum);
    expect(marginal0.entries).toEqual([
      [[1], 0.30000000000000004],
      [[2], 0.7],
    ]);
    expect(marginal0.fields).toEqual([{ kind: "int" }]);

    expect(marginal1.fields).toEqual([
      { kind: "categorical", labels: ["MISS", "HIT"] },
    ]);
    const byValue = new Map(
      marginal1.entries.map(([[value], probability]) => [
        value,
        probability,
      ])
    );
    expect(byValue.get(0)).toBeCloseTo(0.4);
    expect(byValue.get(1)).toBeCloseTo(0.6);
  });
});

describe("tupleData pivot", () => {
  it("builds a grid with joint cells and marginals", () => {
    const pivot = computeTuplePivot(jointIntEnum);
    expect(pivot.xAxis.values).toEqual([1, 2]);
    expect(pivot.yAxis.values).toEqual([0, 1]);
    expect(pivot.cell(1, 0)).toBeCloseTo(0.1);
    expect(pivot.cell(2, 1)).toBeCloseTo(0.4);
    expect(pivot.cell(9, 9)).toBe(0);
    expect(pivot.xMarginal(2)).toBeCloseTo(0.7);
    expect(pivot.yMarginal(1)).toBeCloseTo(0.6);
    expect(pivot.maxCell).toBeCloseTo(0.4);
  });
});

describe("tupleData rows", () => {
  it("sorts by probability descending", () => {
    const rows = computeTupleRows(jointIntEnum, "probability");
    expect(rows[0].values).toEqual([2, 1]);
    expect(rows[0].labels).toEqual(["2", "HIT"]);
    expect(rows.map((r) => r.probability)).toEqual([0.4, 0.3, 0.2, 0.1]);
  });

  it("sorts lexicographically by field", () => {
    const rows = computeTupleRows(jointIntEnum, "lexicographic");
    expect(rows.map((r) => r.values)).toEqual([
      [1, 0],
      [1, 1],
      [2, 0],
      [2, 1],
    ]);
  });
});
