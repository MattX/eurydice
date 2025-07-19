import { Distribution } from "../util";

export interface DistributionData {
  name: string;
  distribution: Distribution;
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

export function generateValuesOnlyCSV(distributions: DistributionData[]): string {
  const allOutcomes = new Set<number>();
  distributions.forEach(({ distribution }) => {
    distribution.probabilities.forEach(([outcome]) => allOutcomes.add(outcome));
  });
  const sortedOutcomes = Array.from(allOutcomes).sort((a, b) => a - b);

  let csv = 'Outcome';
  distributions.forEach(({ name }) => {
    csv += ',' + escapeCSVField(name);
  });
  csv += '\n';

  sortedOutcomes.forEach(outcome => {
    csv += outcome.toString();
    distributions.forEach(({ distribution }) => {
      const prob = distribution.probabilities.find(([o]) => o === outcome)?.[1] || 0;
      csv += ',' + prob.toString();
    });
    csv += '\n';
  });

  return csv;
}

export function generateAnyDiceFormatCSV(distributions: DistributionData[]): string {
  let csv = '';
  
  distributions.forEach(({ name, distribution }, index) => {
    if (index > 0) csv += '\n';
    
    const outcomes = distribution.probabilities.map(([outcome]) => outcome);
    const probabilities = distribution.probabilities.map(([, probability]) => probability);
    
    const mean = outcomes.reduce((sum, val, i) => sum + val * probabilities[i], 0);
    const variance = outcomes.reduce((sum, val, i) => sum + Math.pow(val - mean, 2) * probabilities[i], 0);
    const stdDev = Math.sqrt(variance);
    const min = Math.min(...outcomes);
    const max = Math.max(...outcomes);
    
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