import { describe, expect, it } from "vitest";

import {
  applyTextEdits,
  editorDiagnostics,
  EurydiceDiagnostic,
  isApplicableFix,
  SuggestedFix,
  TextEdit,
} from "./diagnostics";

const edit = (start: number, end: number, replacement: string): TextEdit => ({
  range: { source: 0, range: { start, end } },
  replacement,
});

describe("structured diagnostics", () => {
  it("applies multi-edit fixes without shifting later ranges", () => {
    expect(
      applyTextEdits("output 1 == 1;", [
        edit(10, 11, ""),
        edit(13, 14, ""),
      ]),
    ).toBe("output 1 = 1");
  });
});

describe("isApplicableFix", () => {
  const submitted = { id: 0, name: "submission 1", text: "output 1 == 1" };
  const fix: SuggestedFix = {
    message: "Remove the second `=`",
    edits: [edit(10, 11, "")],
  };

  it("applies a fix to the text its offsets were measured in", () => {
    expect(isApplicableFix(fix, submitted, submitted.text)).toBe(true);
  });

  it("withholds a fix once the editor has moved on", () => {
    // With "run live" off nothing clears the diagnostics as the user types, so
    // the offsets in the fix no longer describe what is in the editor.
    expect(isApplicableFix(fix, submitted, "output 11 == 1")).toBe(false);
    expect(isApplicableFix(fix, null, submitted.text)).toBe(false);
  });

  it("withholds a fix that edits an earlier submission", () => {
    const elsewhere: SuggestedFix = {
      ...fix,
      edits: [{ ...fix.edits[0], range: { source: 1, range: { start: 0, end: 0 } } }],
    };

    expect(isApplicableFix(elsewhere, submitted, submitted.text)).toBe(false);
    expect(isApplicableFix({ ...fix, edits: [] }, submitted, submitted.text)).toBe(false);
  });
});

describe("editorDiagnostics", () => {
  const diagnostic: EurydiceDiagnostic = {
    code: "name.undefined_variable",
    severity: "error",
    summary: "Variable `FOO` is not defined",
    labels: [
      {
        range: { source: 0, range: { start: 7, end: 10 } },
        style: "primary",
      },
      {
        range: { source: 0, range: { start: 11, end: 14 } },
        message: "defined later",
        style: "secondary",
      },
    ],
    help: null,
    fix: null,
    trace: [],
  };

  it("marks only the primary span as the problem", () => {
    const [primary, secondary] = editorDiagnostics([diagnostic], 0, 100);

    expect(primary.severity).toBe("error");
    expect(primary.message).toBe("Variable FOO is not defined");

    // The definition this statement merely runs before is correct code, so it
    // is neither underlined as an error nor told it is undefined.
    expect(secondary.severity).toBe("info");
    expect(secondary.message).toBe("defined later");
  });

  it("ignores labels pointing into another submission", () => {
    expect(editorDiagnostics([diagnostic], 1, 100)).toHaveLength(0);
  });

  it("clamps ranges to a document that has since shrunk", () => {
    const [primary] = editorDiagnostics([diagnostic], 0, 8);

    expect(primary.from).toBe(7);
    expect(primary.to).toBe(8);
  });
});
