import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { editorDiagnostics, EurydiceDiagnostic } from "../diagnostics";
import { DiagnosticCard } from "./EditorPane";

describe("DiagnosticCard", () => {
  it("renders compact useful details without exposing the stable code", () => {
    const diagnostic: EurydiceDiagnostic = {
      code: "type.function_argument",
      severity: "error",
      summary: "`[field INDEX:n of TUPLE:n]` requires a tuple",
      labels: [
        {
          range: { source: 0, range: { start: 20, end: 21 } },
          message: "`TUPLE` is an integer: `1`; expected a tuple",
          style: "primary",
        },
        {
          range: { source: 0, range: { start: 0, end: 0 } },
          style: "secondary",
        },
      ],
      notes: ["Tuple values contain two or more fields."],
      help: "Build a tuple with `[tuple A B]`.",
      fixes: [
        {
          message: "Wrap in a tuple",
          applicability: "suggested",
          edits: [],
        },
      ],
      trace: [
        {
          function: "pick {}",
          call: { source: 0, range: { start: 0, end: 1 } },
          definition: null,
          bindings: [
            {
              name: "I",
              value: {
                shape: "number",
                outcome_type: "int",
                collection_size: null,
                preview: "1",
              },
            },
          ],
        },
      ],
      details: { kind: "type_mismatch" },
      incomplete: false,
    };

    const markup = renderToStaticMarkup(
      <DiagnosticCard
        diagnostic={diagnostic}
        canApplyFix={() => true}
        applyFix={() => undefined}
      />,
    );

    expect(markup).toContain(
      "[field INDEX:n of TUPLE:n]</code> requires a tuple",
    );
    expect(markup).toContain("TUPLE</code> is an integer: <code");
    expect(markup).not.toContain("`");
    expect(markup).not.toContain(">here<");
    expect(markup).toContain("Build a tuple with <code");
    expect(markup).toContain(diagnostic.notes[0]);
    expect(markup).toContain("while calling [pick …] with I = 1");
    expect(markup).toContain("Wrap in a tuple");
    expect(markup).not.toContain(diagnostic.code);
    expect(markup).toContain("p-2");
    expect(markup).toContain("size-5");
    expect(markup).toContain("btn-compact");

    const hoverDiagnostics = editorDiagnostics([diagnostic], 0, 100);
    expect(hoverDiagnostics).toHaveLength(2);
    expect(hoverDiagnostics.every(({ message }) => message ===
      "[field INDEX:n of TUPLE:n] requires a tuple")).toBe(true);
    expect(hoverDiagnostics[0].message).not.toContain(diagnostic.help);
    expect(hoverDiagnostics[0].message).not.toContain("expected a tuple");
  });

  it("supports range-only labels without adding supporting text", () => {
    const diagnostic: EurydiceDiagnostic = {
      code: "name.undefined_variable",
      severity: "error",
      summary: "Variable `POOL` is not defined",
      labels: [
        {
          range: { source: 0, range: { start: 7, end: 11 } },
          style: "primary",
        },
      ],
      notes: [],
      help: "Variable names are uppercase and must be assigned before use.",
      fixes: [],
      trace: [],
      details: {
        kind: "undefined_name",
        name: "POOL",
        namespace: "variable",
        suggestions: [],
      },
      incomplete: false,
    };

    const markup = renderToStaticMarkup(
      <DiagnosticCard
        diagnostic={diagnostic}
        canApplyFix={() => false}
        applyFix={() => undefined}
      />,
    );

    expect(markup).not.toContain("not found in the current scope");
    expect(markup).toContain(
      "Variable names are uppercase and must be assigned before use.",
    );
  });
});
