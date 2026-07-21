import { Distribution, TupleDistribution } from "../util";
import { type NamedDistribution, partitionDistributions } from "./chartData";
import { computeTupleRows, fieldName } from "./tupleData";

export interface DistributionData {
  name: string;
  distribution: Distribution;
}

export interface TupleData {
  name: string;
  distribution: TupleDistribution;
}

export function escapeCSVField(field: string): string {
  // If field contains comma, newline, or double quote, it needs to be quoted
  if (field.includes(',') || field.includes('\n') || field.includes('\r') || field.includes('"')) {
    // Escape double quotes by doubling them
    const escaped = field.replace(/"/g, '""');
    return `"${escaped}"`;
  }
  return field;
}

function generateWideBlock(
  title: string,
  distributions: NamedDistribution[],
  outcomes: number[],
  outcomeLabel: (outcome: number) => string
): string {
  const rows = [
    escapeCSVField(title),
    ["Outcome", ...distributions.map(([name]) => escapeCSVField(name))].join(","),
  ];

  for (const outcome of outcomes) {
    rows.push(
      [
        escapeCSVField(outcomeLabel(outcome)),
        ...distributions.map(([, distribution]) =>
          (
            distribution.probabilities.find(([value]) => value === outcome)?.[1] ?? 0
          ).toString()
        ),
      ].join(",")
    );
  }

  return rows.join("\n");
}

/** One CSV block per tuple: a field-per-column joint table. */
function generateTupleBlock({ name, distribution }: TupleData): string {
  const header = [
    ...distribution.fields.map((schema, i) => escapeCSVField(fieldName(schema, i))),
    "Probability",
  ].join(",");
  const rows = computeTupleRows(distribution, "lexicographic").map((row) =>
    [...row.labels.map(escapeCSVField), row.probability.toString()].join(",")
  );
  return [escapeCSVField(name), header, ...rows].join("\n");
}

export function generateSpreadsheetCSV(
  distributions: DistributionData[],
  tuples: TupleData[] = []
): string {
  const namedDistributions: NamedDistribution[] = distributions.map(
    ({ name, distribution }) => [name, distribution]
  );
  const { sections } = partitionDistributions(namedDistributions);

  const blocks = sections
    .map((section) => {
      if (section.kind === "numeric") {
        const outcomes = Array.from(
          new Set(
            section.distributions.flatMap(([, distribution]) =>
              distribution.probabilities.map(([outcome]) => outcome)
            )
          )
        ).sort((a, b) => a - b);
        return generateWideBlock(
          "Numeric outcomes",
          section.distributions,
          outcomes,
          (outcome) => outcome.toString()
        );
      }

      const probabilityOutcomes = section.group.distributions.flatMap(
        ([, distribution]) =>
          distribution.probabilities.map(([outcome]) => outcome)
      );
      const outcomes = Array.from(
        new Set([
          ...section.group.labels.map((_, index) => index),
          ...probabilityOutcomes,
        ])
      ).sort((a, b) => a - b);
      return generateWideBlock(
        section.group.enumName,
        section.group.distributions,
        outcomes,
        (outcome) => section.group.labels[outcome] ?? outcome.toString()
      );
    });

  return [...blocks, ...tuples.map(generateTupleBlock)].join("\n\n");
}

export function generateAnyDiceFormatCSV(distributions: DistributionData[]): string {
  let csv = '';

  const numericDistributions = distributions.filter(
    ({ distribution }) => distribution.enum_name === undefined
  );

  numericDistributions.forEach(({ name, distribution }, index) => {
    if (index > 0) csv += '\n';

    const outcomes = distribution.probabilities.map(([outcome]) => outcome);
    const probabilities = distribution.probabilities.map(([, probability]) => probability);
    
    const mean = outcomes.reduce((sum, val, i) => sum + val * probabilities[i], 0);
    const variance = outcomes.reduce((sum, val, i) => sum + Math.pow(val - mean, 2) * probabilities[i], 0);
    const stdDev = Math.sqrt(variance);
    const min = outcomes.length > 0 ? Math.min(...outcomes) : 0;
    const max = outcomes.length > 0 ? Math.max(...outcomes) : 0;
    
    csv += `${escapeCSVField(name)},${mean},${stdDev},${min},${max}\n`;
    csv += '#,%\n';
    
    distribution.probabilities.forEach(([outcome, probability]) => {
      csv += `${outcome},${(probability * 100).toFixed(10)}\n`;
    });
  });

  return csv;
}

export function downloadCSV(content: string, filename: string): void {
  const blob = new Blob([content], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(url);
}
