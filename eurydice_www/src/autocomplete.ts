import {
  Completion,
  CompletionSource,
  snippet,
} from "@codemirror/autocomplete";
import type { EditorState } from "@codemirror/state";

export interface PrimitiveMetadata {
  identifier: string;
  signature: string;
  snippet: string;
  documentation: string;
  documentation_url: string;
}

/**
 * Completions for the built-in functions.
 *
 * The primitives are looked up from editor state when the completion runs,
 * because they arrive from the worker after the editor is created. Reading
 * them here keeps the extension itself constant, so a late arrival cannot
 * reconfigure the editor out from under an open completion.
 */
export function primitiveCompletionSource(
  getPrimitives: (state: EditorState) => readonly PrimitiveMetadata[],
): CompletionSource {
  return (context) => {
    if (insideStringOrComment(context.state.doc.sliceString(0, context.pos))) {
      return null;
    }

    const bracketPrefix = context.matchBefore(/\[[a-z ]*$/);
    if (!context.explicit && bracketPrefix === null) {
      return null;
    }

    const insideBracket = bracketPrefix !== null;
    const wordPrefix = insideBracket ? null : context.matchBefore(/[a-z]*$/);
    const from = insideBracket
      ? bracketPrefix.from + 1
      : (wordPrefix?.from ?? context.pos);
    const closingBracket =
      insideBracket &&
      context.state.doc.sliceString(context.pos, context.pos + 1) === "]";

    return {
      from,
      options: getPrimitives(context.state).map((primitive) =>
        primitiveCompletion(
          primitive,
          completionTemplate(primitive, insideBracket, closingBracket),
        ),
      ),
      validFor: insideBracket ? /^[a-z ]*$/ : /^[a-z]*$/,
    };
  };
}

function primitiveCompletion(
  primitive: PrimitiveMetadata,
  template: string,
): Completion {
  return {
    label: primitive.signature.slice(1, -1),
    displayLabel: primitive.signature,
    detail: "built-in",
    type: "function",
    info: () => primitiveCompletionInfo(primitive),
    apply: snippet(template),
  };
}

export function completionTemplate(
  primitive: PrimitiveMetadata,
  insideBracket: boolean,
  closingBracket: boolean,
): string {
  if (!insideBracket) {
    return `[${primitive.snippet}]`;
  }
  return primitive.snippet + (closingBracket ? "" : "]");
}

/**
 * Whether the cursor sits inside a string or a comment.
 *
 * This scans rather than consulting the syntax tree, because the tree cannot
 * help in the case that matters most: a string or comment the user has not
 * finished typing does not parse as one.
 */
function insideStringOrComment(source: string): boolean {
  let state: "code" | "string" | "block_comment" | "line_comment" = "code";
  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];
    if (state === "string") {
      if (character === "\\") {
        index += 1;
      } else if (character === '"') {
        state = "code";
      }
    } else if (state === "block_comment") {
      if (character === "\\") {
        state = "code";
      }
    } else if (state === "line_comment") {
      if (character === "\n" || character === "\r") {
        state = "code";
      }
    } else if (character === '"') {
      state = "string";
    } else if (character === "\\") {
      if (source.slice(index, index + 3) === "\\\\\\") {
        state = "line_comment";
        index += 2;
      } else {
        state = "block_comment";
      }
    }
  }
  return state !== "code";
}

function primitiveCompletionInfo(primitive: PrimitiveMetadata): HTMLElement {
  const container = document.createElement("div");
  const documentation = document.createElement("p");
  documentation.textContent = primitive.documentation;
  container.append(documentation);

  const link = document.createElement("a");
  link.href = primitive.documentation_url;
  link.textContent = "Read the specification";
  link.target = "_blank";
  link.rel = "noreferrer";
  container.append(link);
  return container;
}
