import { StateEffect, StateField, Transaction } from "@codemirror/state";

import { EurydiceDiagnostic } from "../diagnostics";
import { PrimitiveMetadata } from "../autocomplete";

/** Everything the editor's extensions need from React, carried in editor state. */
export interface EditorData {
  diagnostics: readonly EurydiceDiagnostic[];
  sourceId: number | null;
  primitives: readonly PrimitiveMetadata[];
}

export const setEditorData = StateEffect.define<EditorData>();

/**
 * Holds the current run's results for the extensions to read.
 *
 * Sending this through a transaction rather than rebuilding the extensions
 * keeps the configuration constant for the editor's lifetime; reconfiguring
 * CodeMirror mid-session discards its temporary state, including an open
 * completion or an active snippet.
 */
export const editorData = StateField.define<EditorData>({
  create: () => ({ diagnostics: [], sourceId: null, primitives: [] }),
  update(value, transaction) {
    for (const effect of transaction.effects) {
      if (effect.is(setEditorData)) {
        return effect.value;
      }
    }
    return value;
  },
});

/**
 * Whether an update delivered new data for the extensions to read.
 *
 * The lint plugin only schedules a run when the document or its own
 * configuration changes, and `forceLinting` can only force a run that is
 * already scheduled. Diagnostics reach the editor as an effect on an unchanged
 * document, so the linter has to be told through `needsRefresh` that such an
 * update concerns it — without that it never runs for them at all.
 */
export function carriesEditorData(update: {
  transactions: readonly Transaction[];
}): boolean {
  return update.transactions.some((transaction) =>
    transaction.effects.some((effect) => effect.is(setEditorData)),
  );
}
