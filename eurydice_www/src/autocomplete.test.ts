import {
  CompletionContext,
  autocompletion,
  nextSnippetField,
  snippet,
} from "@codemirror/autocomplete";
import { Compartment, EditorState, Transaction } from "@codemirror/state";
import { describe, expect, it } from "vitest";

import {
  PrimitiveMetadata,
  completionTemplate,
  primitiveCompletionSource,
} from "./autocomplete";

const primitive: PrimitiveMetadata = {
  identifier: "highest {} of {}",
  signature: "[highest COUNT:n of POOL:d]",
  snippet: "highest ${COUNT} of ${POOL}",
  documentation: "Sums the highest dice.",
  documentation_url:
    "/help/spec/#highest-countn-of-poold-lowest-countn-of-poold-middle-countn-of-poold",
};

function complete(source: string, explicit = false) {
  const state = EditorState.create({
    doc: source,
  });
  return primitiveCompletionSource([primitive])(
    new CompletionContext(state, source.length, explicit),
  );
}

describe("primitive autocomplete", () => {
  it("opens automatically after a bracket and replaces only the call body", async () => {
    const result = await complete("[");
    expect(result?.from).toBe(1);
    expect(result?.options[0].displayLabel).toBe(primitive.signature);
  });

  it("does not open automatically outside a bracket", async () => {
    expect(await complete("hig")).toBeNull();
    expect((await complete("hig", true))?.from).toBe(0);
  });

  it("wraps explicit completions and preserves an existing closing bracket", () => {
    expect(completionTemplate(primitive, false, false)).toBe(
      "[highest ${COUNT} of ${POOL}]",
    );
    expect(completionTemplate(primitive, true, true)).toBe(
      "highest ${COUNT} of ${POOL}",
    );
  });

  it("keeps snippet parameters active while editor features are reconfigured", () => {
    const completionCompartment = new Compartment();
    let state = EditorState.create({
      extensions: [completionCompartment.of(autocompletion())],
    });
    const editor = {
      get state() {
        return state;
      },
      dispatch(transaction: Transaction) {
        state = transaction.state;
      },
    };
    snippet("[highest ${COUNT} of ${POOL}]")(editor, null, 0, 0);
    expect(state.sliceDoc(state.selection.main.from, state.selection.main.to)).toBe("COUNT");

    editor.dispatch(
      state.update({
        effects: completionCompartment.reconfigure(autocompletion()),
      }),
    );
    editor.dispatch(state.update(state.replaceSelection("2")));
    expect(nextSnippetField(editor)).toBe(true);
    expect(state.sliceDoc(state.selection.main.from, state.selection.main.to)).toBe("POOL");
  });

  it("does not complete inside strings or comments", async () => {
    expect(await complete('print "[', true)).toBeNull();
    expect(await complete("\\ comment [ ", true)).toBeNull();
  });
});
