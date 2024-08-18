import CodeMirror, { EditorView, ViewUpdate } from "@uiw/react-codemirror";
import React, { useEffect } from "react";
import { parser } from "./grammar/eurydice";
import {
  foldNodeProp,
  foldInside,
  indentNodeProp,
  LRLanguage,
  LanguageSupport,
} from "@codemirror/language";
import { githubLight } from "@uiw/codemirror-theme-github";
import { styleTags, tags as t } from "@lezer/highlight";

// import { printTree } from "./printTree";
import { WorkerWrapper } from "./worker-wrapper";
import { linter } from "@codemirror/lint";
import Spinner from "./components/Spinner";
import { Distribution } from "./util";
import OutputPane from "./components/OutputPane";

let worker = new WorkerWrapper(
  new Worker(new URL("./worker.js", import.meta.url))
);

const parserWithMetadata = parser.configure({
  props: [
    styleTags({
      Reference: t.variableName,
      Number: t.number,
      String: t.string,
      Comment: t.blockComment,
      "over output print set named": t.keyword,
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
    commentTokens: { block: { open: "\\", close: "\\" } },
  },
});

const languageSupport = new LanguageSupport(language);

function App() {
  const [editorText, setEditorText] = React.useState("");
  const [output, setOutput] = React.useState(
    new Map<string, Distribution>()
  );
  const [error, setError] = React.useState<EurydiceError|null>(null);
  const [runLive, setRunLiveInner] = React.useState(true);
  const [running, setRunning] = React.useState(false);
  const [printOutputs, setPrintOutputs] = React.useState<[string, string][]>([]);

  function setRunLive(val: boolean) {
    setRunLiveInner(val);
    if (val) {
      localStorage.removeItem("eurydice0_run_live");
      run(editorText);
    } else {
      localStorage.setItem("eurydice0_run_live", "false");
    }
  }

  function attachOnMessage(worker: WorkerWrapper) {
    worker.setOnmessage((event: MessageEvent<EurydiceMessage>) => {
      if (event.data.Err !== undefined) {
        setRunning(false);
        setError(event.data.Err);
      } else if (event.data.Ok !== undefined) {
        setRunning(false);
        setError(null);
        const chartData = new Map<string, Distribution>();
        for (const [key, value] of event.data.Ok!) {
          if (value.Distribution !== undefined) {
            chartData.set(key, value.Distribution);
          } else if (value.Int !== undefined) {
            chartData.set(key, { probabilities: [[value.Int, 1]] });
          } else if (value.List !== undefined) {
            const length = value.List.length;
            chartData.set(key, {
              probabilities: value.List.map((x) => [x, 1.0 / length]),
            });
          }
        }
        setOutput(chartData);
      } else if (event.data.Print !== undefined) {
        setPrintOutputs([...printOutputs, event.data.Print]);
      }
    });
  }

  function run(val?: string) {
    if (running) {
      worker.terminate();
      worker = new WorkerWrapper(
        new Worker(new URL("./worker.js", import.meta.url))
      );
      attachOnMessage(worker);
    }
    setRunning(true);
    setPrintOutputs([]);
    worker.postMessage(val ?? editorText);
  }

  useEffect(() => {
    // Attach the onmessage listener
    attachOnMessage(worker);

    // Load the saved state from local storage
    const savedRunLive = localStorage.getItem("eurydice0_run_live") !== "false";
    if (!savedRunLive) {
      setRunLiveInner(false);
    }
    const savedText =
      localStorage.getItem("eurydice0_editor_program") || "output 1d6 + 2";
    setEditorText(savedText);
    if (savedRunLive) {
      run(savedText);
    }
  }, []);

  const onChange = React.useCallback(
    (val: string, _viewUpdate: ViewUpdate) => {
      setEditorText(val);
      localStorage.setItem("eurydice0_editor_program", val);
      if (runLive) {
        run(val);
      }
    },
    [runLive, running]
  );

  const eurydiceLinter = linter((_view: EditorView) => {
    if (error === null) {
      return [];
    }
    return [{
      // Clamp values here - a slightly delayed worker response can cause
      // a crash if the error is now out of bounds.
      from: Math.min(error.from, editorText.length),
      to: Math.min(error.from, editorText.length),
      message: error.message,
      severity: "error",
    }];
  });

  const runButtonClass = runLive
    ? "border-gray-400 text-gray-400"
    : "border-blue-500 hover:border-blue-700";

  return (
    <>
      <div className="flex flex-col min-h-screen">
        <nav className="w-full md:w-1/2 p-4">
          <ul className="flex [&>*]:border-l [&>*]:border-gray-500 [&>*]:px-4">
            <li className="border-none">
              <a href="#">Eurydice</a>
            </li>
            <li>About</li>
            <li>Help</li>
            <li>
              <a href="https://anydice.com">AnyDice</a>
            </li>
          </ul>
        </nav>
        <div className="flex flex-grow md:min-h-[400px]">
          <div className="flex flex-col md:flex-row w-full h-full items-stretch">
            <div className="w-full md:w-1/2 p-4 border-gray-300 border">
              <div className="flex flex-row mb-4 px-2">
                <label className="border-2 border-blue-500 hover:border-blue-700 py-1 px-2 mr-1 rounded align-middle">
                  <input
                    type="checkbox"
                    name="runLiveCheckbox"
                    checked={runLive}
                    onChange={(e) => setRunLive(e.target.checked)}
                  />{" "}
                  Run live
                </label>
                <button
                  disabled={runLive}
                  onClick={() => !runLive && run()}
                  className={`border-2 ${runButtonClass} py-1 px-2 mx-1 rounded`}
                >
                  Run
                </button>
                {running && <Spinner />}
              </div>
              <CodeMirror
                value={editorText}
                onChange={onChange}
                extensions={[languageSupport, eurydiceLinter]}
                theme={githubLight}
                height="100%"
                minHeight="100%"
              />
            </div>
            <div className="w-full md:w-1/2 p-4 border-gray-300 border">
              <OutputPane distributions={output} error={error?.message} printOutputs={printOutputs} />
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default App;

interface EurydiceMessage {
  Ok: [string, DistributionWrapper][] | undefined;
  Err: EurydiceError | undefined;
  Print: [string, string] | undefined;
}

interface EurydiceError {
  message: string;
  from: number;
  to: number;
}

interface DistributionWrapper {
  Distribution: Distribution | undefined;
  Int: number | undefined;
  List: number[] | undefined;
}
