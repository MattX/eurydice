import { WireDistribution } from "./utils/tupleData";
import type { Diagnostic as CodeMirrorDiagnostic } from "@codemirror/lint";

export interface SourceRange {
  source: number;
  range: { start: number; end: number };
}

export interface DiagnosticLabel {
  range: SourceRange;
  message?: string;
  style: "primary" | "secondary";
}

export interface TextEdit {
  range: SourceRange;
  replacement: string;
}

export interface SuggestedFix {
  message: string;
  applicability: "machine_applicable" | "suggested";
  edits: TextEdit[];
}

export interface ValueSummary {
  shape: string;
  outcome_type: string;
  collection_size: number | null;
  preview: string;
}

export interface EvaluationFrame {
  function: string;
  call: SourceRange;
  definition: SourceRange | null;
  bindings: { name: string; value: ValueSummary }[];
}

export interface EurydiceDiagnostic {
  code: string;
  severity: "error" | "warning";
  summary: string;
  labels: DiagnosticLabel[];
  notes: string[];
  help: string | null;
  fixes: SuggestedFix[];
  trace: EvaluationFrame[];
  details: { kind: string; [key: string]: unknown };
  incomplete: boolean;
}

export interface DiagnosticSource {
  id: number;
  name: string;
  text: string;
}

export interface RunReport {
  outputs: { name: string; distribution: WireDistribution }[];
  diagnostics: EurydiceDiagnostic[];
  sources: DiagnosticSource[];
}

export function applyTextEdits(source: string, edits: TextEdit[]): string {
  return [...edits]
    .sort((left, right) => right.range.range.start - left.range.range.start)
    .reduce(
      (result, edit) =>
        result.slice(0, edit.range.range.start) +
        edit.replacement +
        result.slice(edit.range.range.end),
      source,
    );
}

export function currentSourceId(report: RunReport): number | null {
  return report.sources[report.sources.length - 1]?.id ?? null;
}

export function editorDiagnostics(
  diagnostics: readonly EurydiceDiagnostic[],
  diagnosticSourceId: number | null,
  documentLength: number,
): CodeMirrorDiagnostic[] {
  return diagnostics.flatMap((diagnostic) =>
    diagnostic.labels
      .filter((label) => label.range.source === diagnosticSourceId)
      .map((label) => ({
        // Clamp values here - a slightly delayed worker response can cause
        // a crash if the diagnostic is now out of bounds.
        from: Math.min(label.range.range.start, documentLength),
        to: Math.min(label.range.range.end, documentLength),
        // CodeMirror already supplies severity styling and the source underline.
        // Keep its hover concise; the full explanation lives in the card below.
        message: diagnostic.summary.replace(/`/g, ""),
        severity: diagnostic.severity,
      })),
  );
}
