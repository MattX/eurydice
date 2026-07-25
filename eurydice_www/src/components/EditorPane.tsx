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

export default function EditorPane(props: EditorPaneProps) {
  const isDarkMode = React.useContext(DarkModeContext);

  const eurydiceLinter = linter(() => {
    if (props.error === null) {
      return [];
    }
    return [
      {
        // Clamp values here - a slightly delayed worker response can cause
        // a crash if the error is now out of bounds.
        from: Math.min(props.error.from, props.editorText.length),
        to: Math.min(props.error.to, props.editorText.length),
        message: props.error.message,
        severity: "error",
      },
    ];
  });

  let errorIcon = null;
  if (props.error) {
    errorIcon = (
      <WithTooltip text="This code contains errors. Hover red marks in the editor to see details.">
        <Warning />
      </WithTooltip>
    );
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
          {errorIcon}
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

  error: { from: number; to: number; message: string } | null;
  printOutputs: [string, string][];

  exportButton?: React.ReactNode;
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
