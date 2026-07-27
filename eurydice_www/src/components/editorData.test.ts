import { EditorState } from "@codemirror/state";
import { describe, expect, it } from "vitest";

import { carriesEditorData, editorData, setEditorData } from "./editorData";
import { EurydiceDiagnostic } from "../diagnostics";

const diagnostic: EurydiceDiagnostic = {
  code: "name.undefined_variable",
  severity: "error",
  summary: "Variable `FOO` is not defined",
  labels: [
    { range: { source: 0, range: { start: 7, end: 10 } }, style: "primary" },
  ],
  notes: [],
  help: null,
  fixes: [],
  trace: [],
  incomplete: false,
};

const state = () =>
  EditorState.create({ doc: "output FOO", extensions: [editorData] });

describe("editorData", () => {
  it("starts empty and takes the value of the latest effect", () => {
    const initial = state();
    expect(initial.field(editorData).diagnostics).toEqual([]);

    const updated = initial.update({
      effects: setEditorData.of({
        diagnostics: [diagnostic],
        sourceId: 0,
        primitives: [],
      }),
    }).state;

    expect(updated.field(editorData).diagnostics).toEqual([diagnostic]);
    expect(updated.field(editorData).sourceId).toBe(0);
  });

  it("keeps its value across unrelated transactions", () => {
    const withData = state().update({
      effects: setEditorData.of({
        diagnostics: [diagnostic],
        sourceId: 0,
        primitives: [],
      }),
    }).state;

    const typed = withData.update({
      changes: { from: 10, insert: "D" },
    }).state;

    expect(typed.field(editorData).diagnostics).toEqual([diagnostic]);
  });
});

/**
 * The linter is only scheduled by a document or configuration change, and
 * `forceLinting` can only force a run that is already scheduled. Diagnostics
 * arrive as an effect on an unchanged document, so if this predicate stops
 * recognising them the editor silently shows no marks at all.
 */
describe("carriesEditorData", () => {
  it("recognises the transaction that delivers new diagnostics", () => {
    const transaction = state().update({
      effects: setEditorData.of({
        diagnostics: [diagnostic],
        sourceId: 0,
        primitives: [],
      }),
    });

    expect(carriesEditorData({ transactions: [transaction] })).toBe(true);
  });

  it("ignores ordinary editing", () => {
    const transaction = state().update({ changes: { from: 10, insert: "D" } });

    expect(carriesEditorData({ transactions: [transaction] })).toBe(false);
    expect(carriesEditorData({ transactions: [] })).toBe(false);
  });
});
