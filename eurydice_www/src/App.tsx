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
  const [errors, setErrors] = React.useState<EurydiceError[]>([]);
  const [runLive, setRunLiveInner] = React.useState(true);
  const [running, setRunning] = React.useState(false);

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
      setRunning(false);
      if (event.data.Err) {
        setErrors([event.data.Err]);
      } else {
        setErrors([]);
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
    return errors.map((e) => ({
      // Clamp values here - a slightly delayed worker response can cause
      // a crash if the error is now out of bounds.
      from: Math.min(e.from, editorText.length),
      to: Math.min(e.from, editorText.length),
      message: e.message,
      severity: "error",
    }));
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
              <OutputPane distributions={output} />
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
