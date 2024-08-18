import { githubLight } from "@uiw/codemirror-theme-github";
import { Information, Spinner, Warning } from "./Icons";
import CodeMirror, { EditorView } from "@uiw/react-codemirror";
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

export default function EditorPane(props: EditorPaneProps) {
  const runButtonClass = props.runLive
    ? "border-gray-400 text-gray-400"
    : "border-blue-500 hover:border-blue-700";

  const eurydiceLinter = linter((_view: EditorView) => {
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
      <div>
        <h3 className="font-bold py-2">Print log</h3>
        {outputDivs}
      </div>
    );
    outputIcon = (
      <WithTooltip text="Print output is available below the code editor.">
        <Information />
      </WithTooltip>
    );
  }

  return (
    <>
      <div className="flex flex-row mb-4 px-2">
        <label className="border-2 border-blue-500 hover:border-blue-700 py-1 px-2 mr-1 rounded align-middle">
          <input
            type="checkbox"
            name="runLiveCheckbox"
            checked={props.runLive}
            onChange={(e) => props.setRunLive(e.target.checked)}
          />{" "}
          Run live
        </label>
        <button
          disabled={props.runLive}
          onClick={() => !props.runLive && props.run()}
          className={`border-2 ${runButtonClass} py-1 px-2 mx-1 rounded`}
        >
          Run
        </button>
        {props.running && <Spinner />}
        {outputIcon}
        {errorIcon}
      </div>
      <CodeMirror
        value={props.editorText}
        onChange={props.onChange}
        extensions={[languageSupport, eurydiceLinter]}
        theme={githubLight}
      />
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
      function: t.definitionKeyword,
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
