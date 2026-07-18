import { describe, expect, it } from "vitest";
import { DisplayMode, prepareChartData } from "./chartData";
import { generateAnyDiceFormatCSV, generateValuesOnlyCSV } from "./csvExport";
import { computeTableData } from "./tableData";
import { Distribution } from "../util";

const resultDistribution: Distribution = {
  probabilities: [[0, 0.25], [1, 0.75]],
  enum_name: "RESULT",
  labels: ["MISS", "HIT"],
};

describe("enum output metadata", () => {
  it("uses member names in charts and tables", () => {
    const distributions: [string, Distribution][] = [["attack", resultDistribution]];
    expect(prepareChartData(distributions, DisplayMode.Distribution).labels).toEqual(["MISS", "HIT"]);
    expect(computeTableData(distributions, DisplayMode.Distribution, [0, 1]).map((row) => row.outcomeLabel))
      .toEqual(["MISS", "HIT"]);
  });

  it("exports enum member names without numeric statistics", () => {
    const distributions = [{ name: "attack", distribution: resultDistribution }];
    expect(generateValuesOnlyCSV(distributions)).toContain("MISS,0.25");
    const anyDice = generateAnyDiceFormatCSV(distributions);
    expect(anyDice).toContain("attack,RESULT");
    expect(anyDice).toContain("HIT,75.0000000000");
    expect(anyDice).not.toContain("0.75,0.433");
  });
});
