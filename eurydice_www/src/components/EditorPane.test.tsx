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
      primary_label: {
        range: { source: 0, range: { start: 20, end: 21 } },
        message: "`TUPLE` is an integer: `1`; expected a tuple",
      },
      secondary_labels: [
        {
          range: { source: 0, range: { start: 0, end: 0 } },
        },
      ],
      help: "Tuple values contain two or more fields. Build a tuple with `[tuple A B]`.",
      fix: {
        message: "Wrap in a tuple",
        edits: [],
      },
      trace: [
        {
          function: "pick {}",
          call: { source: 0, range: { start: 0, end: 1 } },
          definition: null,
          bindings: [{ name: "I", value: "1" }],
        },
      ],
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
    expect(markup).toContain("Tuple values contain two or more fields.");
    expect(markup).toContain("Wrap in a tuple");
    expect(markup).not.toContain(diagnostic.code);

    expect(markup).toContain("while calling [pick …] with I = 1");
    expect(markup.match(/while calling/g)).toHaveLength(1);

    // Two labels plus a hover mark built from the trace frame's call site.
    const hoverDiagnostics = editorDiagnostics([diagnostic], 0, 100);
    expect(hoverDiagnostics).toHaveLength(3);
    expect(hoverDiagnostics[2].message).toBe("while calling pick …");
    expect(hoverDiagnostics[2].severity).toBe("info");
    expect(hoverDiagnostics[0].message).toBe(
      "[field INDEX:n of TUPLE:n] requires a tuple",
    );
    expect(hoverDiagnostics[0].message).not.toContain(diagnostic.help);
    expect(hoverDiagnostics[0].message).not.toContain("expected a tuple");
  });

  it("supports range-only labels without adding supporting text", () => {
    const diagnostic: EurydiceDiagnostic = {
      code: "name.undefined_variable",
      severity: "error",
      summary: "Variable `POOL` is not defined",
      primary_label: {
        range: { source: 0, range: { start: 7, end: 11 } },
      },
      secondary_labels: [],
      help: "Variable names are uppercase and must be assigned before use.",
      fix: null,
      trace: [],
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
