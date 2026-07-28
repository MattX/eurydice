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

export interface EvaluationFrame {
  function: string;
  call: SourceRange;
  definition: SourceRange | null;
  bindings: { name: string; value: string }[];
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

/**
 * Whether a fix can still be applied to the text in the editor.
 *
 * An edit is a set of offsets into the submission that produced it. Applying it
 * to anything else — the editor has moved on since, or the fix points into an
 * earlier submission — would land in the wrong place and corrupt the program,
 * so the fix is withheld rather than offered.
 */
export function isApplicableFix(
  fix: SuggestedFix,
  source: DiagnosticSource | null,
  editorText: string,
): boolean {
  return (
    source !== null &&
    editorText === source.text &&
    fix.edits.length > 0 &&
    fix.edits.every((edit) => edit.range.source === source.id)
  );
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

/**
 * The submission a report's diagnostics are anchored to, which the engine
 * always places last.
 */
export function currentSource(report: RunReport): DiagnosticSource | null {
  return report.sources[report.sources.length - 1] ?? null;
}

export function editorDiagnostics(
  diagnostics: readonly EurydiceDiagnostic[],
  diagnosticSourceId: number | null,
  documentLength: number,
): CodeMirrorDiagnostic[] {
  return diagnostics.flatMap((diagnostic) => {
    const marks = diagnostic.labels
      .filter((label) => label.range.source === diagnosticSourceId)
      .map((label) => {
        const isPrimary = label.style === "primary";
        return {
          range: label.range,
          // CodeMirror already supplies severity styling and the source underline.
          // Keep the primary hover concise; the full explanation lives in the
          // card below. A secondary label points somewhere the problem is not,
          // so it says what it is doing there rather than repeating the summary.
          message: isPrimary
            ? diagnostic.summary
            : (label.message ?? diagnostic.summary),
          // Only the primary span is the thing that is wrong. Marking the
          // others as errors would underline, say, a correct definition that a
          // statement merely runs before.
          severity: isPrimary ? diagnostic.severity : ("info" as const),
        };
      });
    // The engine reports call sites as trace frames rather than as labels, so
    // the editor builds its own marks for them.
    const traceMarks = diagnostic.trace
      .filter((frame) => frame.call.source === diagnosticSourceId)
      .map((frame) => ({
        range: frame.call,
        message: `while calling ${frame.function.split("{}").join("…")}`,
        severity: "info" as const,
      }));
    return [...marks, ...traceMarks].map((mark) => ({
      // Clamp values here - a slightly delayed worker response can cause
      // a crash if the diagnostic is now out of bounds.
      from: Math.min(mark.range.range.start, documentLength),
      to: Math.min(mark.range.range.end, documentLength),
      message: stripCode(mark.message),
      severity: mark.severity,
    }));
  });
}

function stripCode(message: string): string {
  return message.replace(/`/g, "");
}
