import { describe, it, expect } from "vitest";
import { TupleDistribution } from "../util";
import {
  normalizeFieldSchema,
  normalizeTupleScalar,
  normalizeTupleSequence,
  normalizeTupleDistribution,
  fieldValueLabel,
  fieldAxis,
  computeMarginals,
  computeTuplePivot,
  computeTupleRows,
} from "./tupleData";

// A 2x2 joint distribution: field 0 is an int (1..2), field 1 is an enum.
const jointIntEnum: TupleDistribution = {
  fields: [
    { kind: "int" },
    { kind: "enum", enumName: "RESULT", labels: ["MISS", "HIT"] },
  ],
  probabilities: [
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
      normalizeFieldSchema({ Enum: { enum_name: "R", labels: ["A", "B"] } })
    ).toEqual({ kind: "enum", enumName: "R", labels: ["A", "B"] });
  });

  it("turns a tuple scalar into a certain outcome", () => {
    const dist = normalizeTupleScalar({ fields: ["Int", "Int"], values: [3, 5] });
    expect(dist.probabilities).toEqual([[[3, 5], 1]]);
    expect(dist.fields).toEqual([{ kind: "int" }, { kind: "int" }]);
  });

  it("turns a tuple sequence into a uniform distribution, collapsing duplicates", () => {
    const dist = normalizeTupleSequence({
      fields: ["Int", "Int"],
      values: [
        [1, 1],
        [1, 1],
        [2, 2],
      ],
    });
    const byKey = new Map(dist.probabilities.map(([o, p]) => [o.join(","), p]));
    expect(byKey.get("1,1")).toBeCloseTo(2 / 3);
    expect(byKey.get("2,2")).toBeCloseTo(1 / 3);
    expect(dist.probabilities).toHaveLength(2);
  });

  it("passes distribution probabilities through", () => {
    const dist = normalizeTupleDistribution({
      fields: ["Int", { Enum: { enum_name: "R", labels: ["A"] } }],
      probabilities: [[[1, 0], 1]],
    });
    expect(dist.fields[1]).toEqual({ kind: "enum", enumName: "R", labels: ["A"] });
  });
});

describe("tupleData labels and axes", () => {
  it("labels enum fields by member name and ints by value", () => {
    expect(fieldValueLabel({ kind: "int" }, 7)).toBe("7");
    expect(
      fieldValueLabel({ kind: "enum", enumName: "R", labels: ["MISS", "HIT"] }, 1)
    ).toBe("HIT");
  });

  it("fills the observed range for int axes", () => {
    expect(fieldAxis({ kind: "int" }, [2, 4]).values).toEqual([2, 3, 4]);
  });

  it("uses every enum member for enum axes", () => {
    const axis = fieldAxis(
      { kind: "enum", enumName: "R", labels: ["A", "B", "C"] },
      [0, 2]
    );
    expect(axis.values).toEqual([0, 1, 2]);
    expect(axis.labels).toEqual(["A", "B", "C"]);
  });
});

describe("tupleData marginals", () => {
  it("sums out other fields", () => {
    const [marginal0, marginal1] = computeMarginals(jointIntEnum);
    expect(marginal0.probabilities).toEqual([
      [1, 0.30000000000000004],
      [2, 0.7],
    ]);
    expect(marginal0.enum_name).toBeUndefined();

    expect(marginal1.enum_name).toBe("RESULT");
    expect(marginal1.labels).toEqual(["MISS", "HIT"]);
    const byValue = new Map(marginal1.probabilities);
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
