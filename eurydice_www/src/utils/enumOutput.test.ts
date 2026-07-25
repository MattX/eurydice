import { describe, expect, it } from "vitest";
import { DisplayMode, prepareChartData } from "./chartData";
import { generateAnyDiceFormatCSV, generateSpreadsheetCSV } from "./csvExport";
import { computeTableData } from "./tableData";
import { ScalarDistribution } from "../util";
import { scalarDistribution } from "./testData";

const resultDistribution: ScalarDistribution = scalarDistribution(
  [[0, 0.25], [1, 0.75]],
  { kind: "enum", enumName: "RESULT", labels: ["MISS", "HIT"] }
);

describe("enum output metadata", () => {
  it("uses member names in charts and tables", () => {
    const distributions: [string, ScalarDistribution][] = [["attack", resultDistribution]];
    expect(prepareChartData(distributions, DisplayMode.Distribution).labels).toEqual(["MISS", "HIT"]);
    expect(computeTableData(distributions, DisplayMode.Distribution, [0, 1]).map((row) => row.outcomeLabel))
      .toEqual(["MISS", "HIT"]);
  });

  it("exports enum member names without numeric statistics", () => {
    const distributions: [string, ScalarDistribution][] = [
      ["attack", resultDistribution],
    ];
    expect(generateSpreadsheetCSV(distributions)).toBe(
      "RESULT\nOutcome,attack\nMISS,0.25\nHIT,0.75"
    );
    expect(generateAnyDiceFormatCSV(distributions)).toBe("");
  });
});
