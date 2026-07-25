import { githubDark, githubLight } from "@uiw/codemirror-theme-github";
import { Information, Share, Spinner, Warning } from "./Icons";
import CodeMirror from "@uiw/react-codemirror";
import { linter } from "@codemirror/lint";
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
  EurydiceDiagnostic,
  SuggestedFix,
} from "../diagnostics";

export default function EditorPane(props: EditorPaneProps) {
  const isDarkMode = React.useContext(DarkModeContext);

  const eurydiceLinter = linter(() => {
    return props.diagnostics.flatMap((diagnostic) =>
      diagnostic.labels
        .filter((label) => label.range.source === props.diagnosticSourceId)
        .map((label) => ({
          // Clamp values here - a slightly delayed worker response can cause
          // a crash if the diagnostic is now out of bounds.
          from: Math.min(label.range.range.start, props.editorText.length),
          to: Math.min(label.range.range.end, props.editorText.length),
          message: [
            diagnostic.summary,
            label.message || null,
            diagnostic.help,
          ].filter(Boolean).join(" — "),
          severity: diagnostic.severity,
        })),
    );
  });

  const errorCount = props.diagnostics.filter(
    (diagnostic) => diagnostic.severity === "error",
  ).length;
  const warningCount = props.diagnostics.length - errorCount;
  let diagnosticIcon = null;
  if (errorCount > 0) {
    diagnosticIcon = (
      <WithTooltip text={`${errorCount} error${errorCount === 1 ? "" : "s"}. See the diagnostics below the editor.`}>
        <Warning color="var(--danger)" />
      </WithTooltip>
    );
  } else if (warningCount > 0) {
    diagnosticIcon = (
      <WithTooltip text={`${warningCount} warning${warningCount === 1 ? "" : "s"}. See the diagnostics below the editor.`}>
        <Warning color="var(--warning)" />
      </WithTooltip>
    );
  }

  function applyFix(fix: SuggestedFix) {
    if (
      props.diagnosticSourceId === null ||
      fix.edits.some((edit) => edit.range.source !== props.diagnosticSourceId)
    ) {
      return;
    }
    props.onChange(applyTextEdits(props.editorText, fix.edits));
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
          extensions={[languageSupport, eurydiceLinter]}
          theme={isDarkMode ? githubDark : githubLight}
        />
      </div>
      {props.diagnostics.length > 0 && (
        <div className="mt-3 space-y-2" aria-label="Diagnostics">
          {props.diagnostics.map((diagnostic, diagnosticIndex) => (
            <DiagnosticCard
              key={`${diagnostic.code}-${diagnosticIndex}`}
              diagnostic={diagnostic}
              canApplyFix={(fix) =>
                props.diagnosticSourceId !== null &&
                fix.edits.every(
                  (edit) => edit.range.source === props.diagnosticSourceId,
                )
              }
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
  diagnosticSourceId: number | null;
  printOutputs: [string, string][];

  exportButton?: React.ReactNode;
}

function DiagnosticCard({
  diagnostic,
  canApplyFix,
  applyFix,
}: {
  diagnostic: EurydiceDiagnostic;
  canApplyFix: (fix: SuggestedFix) => boolean;
  applyFix: (fix: SuggestedFix) => void;
}) {
  const color = diagnostic.severity === "error" ? "var(--danger)" : "var(--warning)";
  return (
    <section
      className="rounded-lg border p-3 text-sm"
      style={{ borderLeftColor: color, borderLeftWidth: 3, background: "var(--surface-2)" }}
    >
      <div className="flex items-start gap-2">
        <Warning color={color} />
        <div className="min-w-0 grow">
          <div className="font-semibold">{diagnostic.summary}</div>
          <div className="mt-0.5 font-mono text-xs text-[var(--text-muted)]">
            {diagnostic.code}
          </div>
          {diagnostic.help && <p className="mt-2">{diagnostic.help}</p>}
          {diagnostic.notes.map((note, index) => (
            <p className="mt-1 text-[var(--text-muted)]" key={index}>
              {note}
            </p>
          ))}
          {diagnostic.trace.length > 0 && (
            <div className="mt-2 space-y-1 font-mono text-xs">
              {diagnostic.trace.map((frame, index) => {
                const bindings = frame.bindings
                  .map((binding) => `${binding.name} = ${binding.value.preview}`)
                  .join(", ");
                return (
                  <div key={index}>
                    while calling [{frame.function.split("{}").join("…")}]
                    {bindings && ` with ${bindings}`}
                  </div>
                );
              })}
            </div>
          )}
          {diagnostic.fixes.filter(canApplyFix).map((fix, index) => (
            <button
              type="button"
              className="btn btn-secondary mt-2 mr-2"
              onClick={() => applyFix(fix)}
              key={index}
            >
              {fix.message}
            </button>
          ))}
        </div>
      </div>
    </section>
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
