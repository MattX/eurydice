import React, { useEffect } from "react";

import { WorkerWrapper } from "./worker-wrapper";
import { Distribution } from "./util";
import OutputPane from "./components/OutputPane";
import EditorPane from "./components/EditorPane";
import Tutorial from "./components/Tutorial";
import { ExternalWebsite } from "./components/Icons";

let worker = new WorkerWrapper(
  new Worker(new URL("./worker.js", import.meta.url)),
);

function App() {
  const [editorText, setEditorText] = React.useState("");
  const [output, setOutput] = React.useState<[string, Distribution][]>([]);
  const [error, setError] = React.useState<EurydiceError | null>(null);
  const [runLive, setRunLiveInner] = React.useState(true);
  const [running, setRunning] = React.useState(false);
  const [printOutputs, setPrintOutputs] = React.useState<[string, string][]>(
    [],
  );
  const [showTutorial, setShowTutorial] = React.useState(false);

  function setRunLive(val: boolean) {
    setRunLiveInner(val);
    if (val) {
      localStorage.removeItem("eurydice0_run_live");
      run(editorText);
    } else {
      localStorage.setItem("eurydice0_run_live", "false");
    }
  }

  function attachOnMessage(
    worker: WorkerWrapper,
    printOutputs: [string, string][] = [],
  ) {
    // Why take in printOutputs as an argument, instead of using the state?
    // Worker messages may be received in quick successiom, and setting React state
    // is asynchronous. This means that the state may not be updated when the next
    // message comes, leading to random messages being dropped.
    // This also means we need to reattach the onmessage listener every time the
    // printOutputs state is updated.
    worker.setOnmessage((event: MessageEvent<EurydiceMessage>) => {
      if (event.data.Err !== undefined) {
        setRunning(false);
        setError(event.data.Err);
      } else if (event.data.Ok !== undefined) {
        setRunning(false);
        setError(null);
        const chartData = new Array();
        for (const [key, value] of event.data.Ok!) {
          if (value.Distribution !== undefined) {
            chartData.push([key, value.Distribution]);
          } else if (value.Int !== undefined) {
            chartData.push([key, { probabilities: [[value.Int, 1]] }]);
          } else if (value.List !== undefined) {
            const length = value.List.length;
            chartData.push([
              key,
              {
                probabilities: value.List.map((x) => [x, 1.0 / length]),
              },
            ]);
          }
        }
        setOutput(chartData);
      } else if (event.data.Print !== undefined) {
        const newPrintOutputs = [...printOutputs, event.data.Print];
        setPrintOutputs(newPrintOutputs);
        attachOnMessage(worker, newPrintOutputs);
      }
    });
  }

  function run(val?: string) {
    if (running) {
      worker.terminate();
      worker = new WorkerWrapper(
        new Worker(new URL("./worker.js", import.meta.url)),
      );
      attachOnMessage(worker);
    }
    setRunning(true);
    setPrintOutputs([]);
    setError(null);
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

  function onChange(val: string) {
    setEditorText(val);
    localStorage.setItem("eurydice0_editor_program", val);
    if (runLive) {
      run(val);
    }
  }

  const tutorial = showTutorial ? (
    <Tutorial
      setEditorText={onChange}
      closeTutorial={() => setShowTutorial(false)}
    />
  ) : null;

  return (
    <>
      <div className="flex flex-col min-h-screen">
        <nav className="w-full md:w-1/2 p-4">
          <ul className="flex [&>*]:border-l [&>*]:border-gray-500 [&>*]:px-4">
            <li className="border-none">
              <a className="hover:underline" href="#">
                Eurydice
              </a>
            </li>
            <li>About</li>
            <li>
              <a
                className="hover:underline"
                href="#"
                onClick={() => setShowTutorial(true)}
              >
                Tutorial
              </a>
            </li>
            <li>
              <a className="hover:underline" href="https://anydice.com">
                AnyDice <ExternalWebsite />
              </a>
            </li>
            <li>
              <a className="hover:underline" href="https://anydice.com/docs">
                AnyDice Documentation <ExternalWebsite />
              </a>
            </li>
          </ul>
        </nav>
        <div className="flex flex-grow md:min-h-[400px]">
          <div className="flex flex-col md:flex-row w-full h-full items-stretch">
            <div className="w-full md:w-1/2 p-4 border-gray-300 border">
              {tutorial}
              <EditorPane
                editorText={editorText}
                onChange={onChange}
                runLive={runLive}
                setRunLive={setRunLive}
                running={running}
                run={() => run()}
                error={error}
                printOutputs={printOutputs}
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
