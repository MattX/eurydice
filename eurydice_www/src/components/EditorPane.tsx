import { githubDark, githubLight } from "@uiw/codemirror-theme-github";
import { Information, Share, Spinner, Warning } from "./Icons";
import CodeMirror from "@uiw/react-codemirror";
import { forceLinting, linter } from "@codemirror/lint";
import {
  autocompletion,
  nextSnippetField,
  prevSnippetField,
} from "@codemirror/autocomplete";
import { Prec } from "@codemirror/state";
import { EditorView, keymap } from "@codemirror/view";
import { styleTags, tags as t } from "@lezer/highlight";
import { parser } from "../grammar/eurydice";
import {
  foldInside,
  foldNodeProp,
  indentNodeProp,
  LanguageSupport,
  LRLanguage,
} from "@codemirror/language";
import WithTooltip from "./Tooltip";
import { DarkModeContext } from "./DarkModeContext";
import React from "react";
import { toast } from "react-hot-toast";
import {
  applyTextEdits,
  DiagnosticSource,
  editorDiagnostics,
  EurydiceDiagnostic,
  isApplicableFix,
  SuggestedFix,
} from "../diagnostics";
import { PrimitiveMetadata, primitiveCompletionSource } from "../autocomplete";
import { carriesEditorData, editorData, setEditorData } from "./editorData";

// @uiw/react-codemirror uses this prop's identity as a signal to reconfigure
// the entire editor. Keep it stable across React renders so accepting a
// completion doesn't immediately remove CodeMirror's temporary snippet state.
const editorBasicSetup = { autocompletion: false } as const;

const snippetTabKeymap = Prec.highest(
  keymap.of([
    {
      key: "Tab",
      run: nextSnippetField,
      shift: prevSnippetField,
    },
  ]),
);

export default function EditorPane(props: EditorPaneProps) {
  const isDarkMode = React.useContext(DarkModeContext);
  const editorViewRef = React.useRef<EditorView | null>(null);
  const captureEditorView = React.useCallback((view: EditorView) => {
    editorViewRef.current = view;
  }, []);
  React.useEffect(() => {
    const view = editorViewRef.current;
    if (view === null) {
      return;
    }
    view.dispatch({
      effects: setEditorData.of({
        diagnostics: props.diagnostics,
        sourceId: props.diagnosticSource?.id ?? null,
        primitives: props.primitives,
      }),
    });
    // Run the linter now rather than after its debounce; the diagnostics are
    // already computed, so there is nothing to wait for.
    forceLinting(view);
  }, [props.diagnostics, props.diagnosticSource, props.primitives]);

  const errorCount = props.diagnostics.filter(
    (diagnostic) => diagnostic.severity === "error",
  ).length;
  const warningCount = props.diagnostics.length - errorCount;
  const plural = (count: number, noun: string) =>
    `${count} ${noun}${count === 1 ? "" : "s"}`;
  let diagnosticIcon = null;
  if (errorCount > 0) {
    diagnosticIcon = (
      <WithTooltip
        text={`${plural(errorCount, "error")}. See the diagnostics below the editor.`}
      >
        <Warning color="var(--danger)" />
      </WithTooltip>
    );
  } else if (warningCount > 0) {
    diagnosticIcon = (
      <WithTooltip
        text={`${plural(warningCount, "warning")}. See the diagnostics below the editor.`}
      >
        <Warning color="var(--warning)" />
      </WithTooltip>
    );
  }

  function canApplyFix(fix: SuggestedFix) {
    return isApplicableFix(fix, props.diagnosticSource, props.editorText);
  }

  function applyFix(fix: SuggestedFix) {
    if (canApplyFix(fix)) {
      props.onChange(applyTextEdits(props.editorText, fix.edits));
    }
  }

  let outputs = null;
  let outputIcon = null;
  if (props.printOutputs.length > 0) {
    const outputDivs = props.printOutputs.map(([value, name], index) => {
      const prefix = name.length > 0 ? name + ": " : "";
      return (
        <div key={index} className="flex flex-row font-mono text-sm">
          <div>
            {prefix}
            {value}
          </div>
        </div>
      );
    });
    outputs = (
      <div className="mt-4 rounded-lg border p-3" style={{ background: "var(--surface-2)" }}>
        <h3 className="mb-2 text-xs font-semibold tracking-wide uppercase text-[var(--text-muted)]">
          Print log
        </h3>
        <div className="space-y-0.5">{outputDivs}</div>
      </div>
    );
    outputIcon = (
      <WithTooltip text="Print output is available below the code editor.">
        <Information />
      </WithTooltip>
    );
  }

  function share() {
    const url = new URL(window.location.href);
    url.hash = `#p=${encodeURIComponent(props.editorText)}`;
    navigator.clipboard.writeText(url.toString());
    toast.success("Link copied to clipboard");
  }

  return (
    <>
      <div className="mb-3 flex flex-row flex-wrap items-center gap-2 clear-both">
        <button className="btn btn-secondary" onClick={share}>
          <Share /> Share
        </button>
        <button
          className="btn-toggle"
          aria-pressed={props.runLive}
          onClick={() => props.setRunLive(!props.runLive)}
        >
          Run live
        </button>
        <button
          disabled={props.runLive}
          onClick={() => !props.runLive && props.run()}
          className="btn btn-primary"
        >
          Run
        </button>
        <div className="ml-auto flex items-center gap-2">
          {props.running && <Spinner />}
          {outputIcon}
          {diagnosticIcon}
          {props.exportButton}
        </div>
      </div>
      <div
        className="overflow-hidden rounded-lg border"
        style={{ borderColor: "var(--border)" }}
      >
        <CodeMirror
          value={props.editorText}
          onChange={props.onChange}
          onCreateEditor={captureEditorView}
          extensions={extensions}
          basicSetup={editorBasicSetup}
          theme={isDarkMode ? githubDark : githubLight}
        />
      </div>
      {props.diagnostics.length > 0 && (
        <div className="mt-3 space-y-2" aria-label="Diagnostics">
          {props.diagnostics.map((diagnostic, diagnosticIndex) => (
            <DiagnosticCard
              key={`${diagnostic.code}-${diagnosticIndex}`}
              diagnostic={diagnostic}
              canApplyFix={canApplyFix}
              applyFix={applyFix}
            />
          ))}
        </div>
      )}
      {outputs}
    </>
  );
}

export interface EditorPaneProps {
  editorText: string;
  onChange: (editorText: string) => void;

  runLive: boolean;
  setRunLive: (runLive: boolean) => void;

  running: boolean;
  run: () => void;

  diagnostics: EurydiceDiagnostic[];
  diagnosticSource: DiagnosticSource | null;
  primitives: PrimitiveMetadata[];
  printOutputs: [string, string][];

  exportButton?: React.ReactNode;
}

export function DiagnosticCard({
  diagnostic,
  canApplyFix,
  applyFix,
}: {
  diagnostic: EurydiceDiagnostic;
  canApplyFix: (fix: SuggestedFix) => boolean;
  applyFix: (fix: SuggestedFix) => void;
}) {
  const color = diagnostic.severity === "error" ? "var(--danger)" : "var(--warning)";
  // Bound so the guard below narrows it for the click handler.
  const fix = diagnostic.fix;
  const supportingMessages = Array.from(
    new Set(
      diagnostic.labels
        .map((label) => label.message?.trim() ?? "")
        .filter(
          (message) =>
            message.length > 0 &&
            message !== diagnostic.summary &&
            message !== diagnostic.help,
        ),
    ),
  );
  return (
    <section
      className="rounded-lg border p-2 text-sm"
      style={{ borderLeftColor: color, borderLeftWidth: 3, background: "var(--surface-2)" }}
    >
      <div className="flex items-start gap-2">
        <Warning color={color} className="mt-0.5 size-5" />
        <div className="min-w-0 grow">
          <div className="text-sm font-medium leading-5">
            <DiagnosticText>{diagnostic.summary}</DiagnosticText>
          </div>
          {(supportingMessages.length > 0 ||
            diagnostic.help ||
            diagnostic.trace.length > 0) && (
            <div className="mt-1 space-y-0.5 text-xs leading-5 text-(--text-muted)">
              {supportingMessages.map((message, index) => (
                <p key={`label-${index}`}>
                  <DiagnosticText>{message}</DiagnosticText>
                </p>
              ))}
              {diagnostic.help && (
                <p>
                  <DiagnosticText>{diagnostic.help}</DiagnosticText>
                </p>
              )}
              {diagnostic.trace.map((frame, index) => {
                const bindings = frame.bindings
                  .map((binding) => `${binding.name} = ${binding.value}`)
                  .join(", ");
                return (
                  <div className="font-mono" key={index}>
                    while calling [{frame.function.split("{}").join("…")}]
                    {bindings && ` with ${bindings}`}
                  </div>
                );
              })}
            </div>
          )}
          {fix && canApplyFix(fix) && (
            <button
              type="button"
              className="btn btn-compact btn-secondary mt-1.5 mr-1.5"
              onClick={() => applyFix(fix)}
            >
              <DiagnosticText>{fix.message}</DiagnosticText>
            </button>
          )}
        </div>
      </div>
    </section>
  );
}

/**
 * Renders a diagnostic string, showing its backtick spans as code.
 *
 * Backticks are the only markup the engine emits, so this deliberately does
 * not pull in a markdown parser. An unpaired backtick stays literal text.
 */
function DiagnosticText({ children }: { children: string }) {
  return (
    <>
      {children.split(/`([^`]*)`/).map((part, index) =>
        index % 2 === 0 ? (
          part
        ) : (
          <code
            className="rounded bg-(--surface-3) px-1 py-0.5 font-mono text-[0.9em] text-[var(--text)]"
            key={index}
          >
            {part}
          </code>
        ),
      )}
    </>
  );
}

const parserWithMetadata = parser.configure({
  props: [
    styleTags({
      Reference: t.variableName,
      Number: t.number,
      String: t.string,
      Comment: t.blockComment,
      LineComment: t.lineComment,
      "over output print set to named": t.keyword,
      "if else loop result": t.controlKeyword,
      "function enum": t.definitionKeyword,
      "( )": t.paren,
      "{ }": t.brace,
      "[ ]": t.squareBracket,
      // Some of these are missing otherwise I get an error: !, /, *, !=, @
      "# - ^ + = < <= > >= & |": t.operator,
      d: t.operatorKeyword,
      "ty-n ty-s": t.typeName,
    }),
    indentNodeProp.add({
      Block: (context) =>
        context.column(context.node.parent?.from ?? 0) + context.unit,
    }),
    foldNodeProp.add({
      Block: foldInside,
    }),
  ],
});

const language = LRLanguage.define({
  parser: parserWithMetadata,
  languageData: {
    commentTokens: { block: { open: "\\", close: "\\" }, line: "\\\\\\" },
  },
});

const languageSupport = new LanguageSupport(language);

const extensions = [
  languageSupport,
  snippetTabKeymap,
  editorData,
  linter(
    (view) => {
      const { diagnostics, sourceId } = view.state.field(editorData);
      return editorDiagnostics(diagnostics, sourceId, view.state.doc.length);
    },
    {
      // The lint plugin only schedules a run when the document or its own
      // configuration changes, and `forceLinting` can only force a run that is
      // already scheduled. Diagnostics reach us as an effect on an unchanged
      // document, so without this the linter would never run for them.
      needsRefresh: carriesEditorData,
    },
  ),
  autocompletion({
    override: [
      primitiveCompletionSource((state) => state.field(editorData).primitives),
    ],
  }),
];
