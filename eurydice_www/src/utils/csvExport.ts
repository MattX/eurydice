import {
  NamedDistribution,
  NamedScalarDistribution,
  isNamedScalarDistribution,
} from "../util";
import { categoricalOutcomes, categoricalProbabilities } from "./chartData";
import { computeTupleRows, fieldName } from "./tupleData";

export function escapeCSVField(field: string): string {
  // If field contains comma, newline, or double quote, it needs to be quoted
  if (
    field.includes(",") ||
    field.includes("\n") ||
    field.includes("\r") ||
    field.includes('"')
  ) {
    // Escape double quotes by doubling them
    const escaped = field.replace(/"/g, '""');
    return `"${escaped}"`;
  }
  return field;
}

function generateWideBlock(
  title: string,
  distributions: NamedScalarDistribution[],
  outcomes: number[],
  outcomeLabel: (outcome: number) => string,
): string {
  const rows = [
    escapeCSVField(title),
    ["Outcome", ...distributions.map(([name]) => escapeCSVField(name))].join(
      ",",
    ),
  ];

  for (const outcome of outcomes) {
    rows.push(
      [
        escapeCSVField(outcomeLabel(outcome)),
        ...distributions.map(([, distribution]) => {
          const index = distribution.values.findIndex(
            (value) => value === outcome,
          );
          return (
            index >= 0 ? distribution.probabilities[index] : 0
          ).toString();
        }),
      ].join(","),
    );
  }

  return rows.join("\n");
}

function generateCategoricalWideBlock(
  distributions: NamedScalarDistribution[],
): string {
  const probabilities = distributions.map(([, distribution]) =>
    categoricalProbabilities(distribution),
  );
  const rows = [
    "Outcomes",
    ["Outcome", ...distributions.map(([name]) => escapeCSVField(name))].join(
      ",",
    ),
  ];
  for (const { key, label } of categoricalOutcomes(distributions)) {
    rows.push(
      [
        escapeCSVField(label),
        ...probabilities.map((distribution) =>
          (distribution.get(key) ?? 0).toString(),
        ),
      ].join(","),
    );
  }
  return rows.join("\n");
}

/** One CSV block per tuple: a field-per-column joint table. */
function generateTupleBlock([name, distribution]: NamedDistribution): string {
  const header = [
    ...distribution.fields.map((_, index) =>
      escapeCSVField(fieldName(distribution, index)),
    ),
    "Probability",
  ].join(",");
  const rows = computeTupleRows(distribution, "lexicographic").map((row) =>
    [...row.labels.map(escapeCSVField), row.probability.toString()].join(","),
  );
  return [escapeCSVField(name), header, ...rows].join("\n");
}

export function generateSpreadsheetCSV(outputs: NamedDistribution[]): string {
  const distributions = outputs.filter(isNamedScalarDistribution);
  const tuples = outputs.filter(
    ([, distribution]) => distribution.fields.length > 1,
  );
  const hasSymbols = distributions.some(
    ([, distribution]) => distribution.fields[0].schema.kind === "categorical",
  );
  const blocks: string[] = [];
  if (distributions.length > 0 && hasSymbols) {
    blocks.push(generateCategoricalWideBlock(distributions));
  } else if (distributions.length > 0) {
    const outcomes = Array.from(
      new Set(distributions.flatMap(([, distribution]) => distribution.values)),
    ).sort((a, b) => a - b);
    blocks.push(
      generateWideBlock(
        "Numeric outcomes",
        distributions,
        outcomes,
        (outcome) => outcome.toString(),
      ),
    );
  }

  return [...blocks, ...tuples.map(generateTupleBlock)].join("\n\n");
}

export function generateAnyDiceFormatCSV(outputs: NamedDistribution[]): string {
  let csv = "";

  const numericDistributions = outputs
    .filter(isNamedScalarDistribution)
    .filter(([, distribution]) => distribution.fields[0].schema.kind === "int");

  numericDistributions.forEach(([name, distribution], index) => {
    if (index > 0) csv += "\n";

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
    const min = outcomes.length > 0 ? Math.min(...outcomes) : 0;
    const max = outcomes.length > 0 ? Math.max(...outcomes) : 0;

    csv += `${escapeCSVField(name)},${mean},${stdDev},${min},${max}\n`;
    csv += "#,%\n";

    distribution.values.forEach((outcome, outcomeIndex) => {
      const probability = distribution.probabilities[outcomeIndex];
      csv += `${outcome},${(probability * 100).toFixed(10)}\n`;
    });
  });

  return csv;
}

export function downloadCSV(content: string, filename: string): void {
  const blob = new Blob([content], { type: "text/csv" });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(url);
}
