import { describe, expect, it } from "vitest";
import { prepareCategoricalChartData } from "./chartData";
import { generateAnyDiceFormatCSV, generateSpreadsheetCSV } from "./csvExport";
import { computeCategoricalTableData } from "./tableData";
import { ScalarDistribution } from "../util";
import { scalarDistribution } from "./testData";

const resultDistribution: ScalarDistribution = scalarDistribution(
  [[0, 0.25], [1, 0.75]],
  { kind: "categorical", labels: ["MISS", "HIT"] }
);

describe("enum output metadata", () => {
  it("uses member names in charts and tables", () => {
    const distributions: [string, ScalarDistribution][] = [["attack", resultDistribution]];
    expect(prepareCategoricalChartData(distributions).labels).toEqual(["MISS", "HIT"]);
    expect(computeCategoricalTableData(distributions).map((row) => row.outcomeLabel))
      .toEqual(["MISS", "HIT"]);
  });

  it("exports enum member names without numeric statistics", () => {
    const distributions: [string, ScalarDistribution][] = [
      ["attack", resultDistribution],
    ];
    expect(generateSpreadsheetCSV(distributions)).toBe(
      "Outcomes\nOutcome,attack\nMISS,0.25\nHIT,0.75"
    );
    expect(generateAnyDiceFormatCSV(distributions)).toBe("");
  });
});
