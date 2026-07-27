import { CompletionContext } from "@codemirror/autocomplete";
import { EditorState } from "@codemirror/state";
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
  return primitiveCompletionSource(() => [primitive])(
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

  // The primitives arrive from the worker after the editor is built, so the
  // source has to read them when it runs rather than when it is created.
  it("reads the primitives available when the completion runs", async () => {
    let available: PrimitiveMetadata[] = [];
    const state = EditorState.create({ doc: "[" });
    const source = primitiveCompletionSource(() => available);

    expect((await source(new CompletionContext(state, 1, false)))?.options).toHaveLength(0);

    available = [primitive];
    expect((await source(new CompletionContext(state, 1, false)))?.options).toHaveLength(1);
  });

  it("does not complete inside strings or comments", async () => {
    expect(await complete('print "[', true)).toBeNull();
    expect(await complete("\\ comment [ ", true)).toBeNull();
  });
});
