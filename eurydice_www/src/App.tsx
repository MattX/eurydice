import React, { useEffect } from "react";

import { WorkerWrapper } from "./worker-wrapper";
import { Distribution } from "./util";
import OutputPane from "./components/OutputPane";
import EditorPane from "./components/EditorPane";
import Tutorial from "./components/Tutorial";
import { ExternalWebsite, Octocat } from "./components/Icons";
import {
  DarkModeContext,
  DarkModeSwitcher,
} from "./components/DarkModeSwitcher";
import EurydiceWorker from "./worker?worker";
import { Toaster } from "react-hot-toast";

let worker = new WorkerWrapper(new EurydiceWorker());

export default function App() {
  return (
    <DarkModeSwitcher>
      <AppInner />
    </DarkModeSwitcher>
  );
}

function AppInner() {
  const [editorText, setEditorText] = React.useState("");
  const [output, setOutput] = React.useState<[string, Distribution][]>([]);
  const [error, setError] = React.useState<EurydiceError | null>(null);
  const [runLive, setRunLiveInner] = React.useState(true);
  const [running, setRunning] = React.useState(false);
  const [printOutputs, setPrintOutputs] = React.useState<[string, string][]>(
    [],
  );
  const [showTutorial, setShowTutorial] = React.useState(false);
  const [isMenuOpen, setIsMenuOpen] = React.useState(false);

  const borderColor = React.useContext(DarkModeContext)
    ? "border-gray-700"
    : "border-gray-300";

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
    printOutputs: [string, string][],
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
        const chartData: [string, Distribution][] = [];
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

        // Check if range is too large
        let minValue = Infinity;
        let maxValue = -Infinity;
        for (const [_, distribution] of chartData) {
          for (const [value] of distribution.probabilities) {
            minValue = Math.min(minValue, value);
            maxValue = Math.max(maxValue, value);
          }
        }

        const range = maxValue - minValue;
        if (range >= 5000) {
          setOutput([]);
          setError({
            message: `Range of outcomes (${range}) is too large to display. Maximum range is 5000.`,
            from: 0,
            to: 0
          });
        } else {
          setOutput(chartData);
        }
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
      worker = new WorkerWrapper(new EurydiceWorker());
    }
    setRunning(true);
    setPrintOutputs([]);
    attachOnMessage(worker, []);
    setError(null);
    worker.postMessage(val ?? editorText);
  }

  useEffect(() => {
    // Attach the onmessage listener
    attachOnMessage(worker, []);

    // Load the saved state from local storage
    const savedRunLive = localStorage.getItem("eurydice0_run_live") !== "false";
    if (!savedRunLive) {
      setRunLiveInner(false);
    }

    // If there is a hash (shared link), use that
    const hash = window.location.hash;
    let savedText: string | null = null;
    if (hash.startsWith("#p=")) {
      savedText = decodeURIComponent(hash.slice(3));
    } else {
      savedText = localStorage.getItem("eurydice0_editor_program") || "output 1d6 + 2";
    }
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
      <Octocat />
      <div><Toaster /></div>
      <div className="flex flex-col min-h-screen">
        <nav className="w-[calc(100%-60px)] p-4">
          <div className="md:hidden flex justify-between items-center">
            <a className="hover:underline" href="#">
              Eurydice
            </a>
            <button
              className="flex flex-col justify-center items-center w-6 h-6 space-y-1"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              aria-label="Toggle menu"
            >
              <span className={`block w-5 h-0.5 bg-current transform transition ${isMenuOpen ? 'rotate-45 translate-y-1.5' : ''}`}></span>
              <span className={`block w-5 h-0.5 bg-current transition ${isMenuOpen ? 'opacity-0' : ''}`}></span>
              <span className={`block w-5 h-0.5 bg-current transform transition ${isMenuOpen ? '-rotate-45 -translate-y-1.5' : ''}`}></span>
            </button>
          </div>
          <ul className={`${isMenuOpen ? 'flex' : 'hidden'} md:flex flex-col md:flex-row flex-wrap mt-4 md:mt-0 *:border-l-0 md:*:border-l *:border-gray-500 *:px-0 md:*:px-4 *:py-2 md:*:py-0`}>
            <li className="border-none hidden md:block">
              <a className="hover:underline" href="#" onClick={() => setIsMenuOpen(false)}>
                Eurydice
              </a>
            </li>
            <li>
              <a className="hover:underline block" href="about/" onClick={() => setIsMenuOpen(false)}>
                About
              </a>
            </li>
            <li>
              <a
                className="hover:underline block"
                href="#"
                onClick={() => {
                  setIsMenuOpen(false);
                  if (confirm("Opening the tutorial will clear the current code. Continue?")) {
                    setShowTutorial(true);
                  }
                }}
              >
                Tutorial
              </a>
            </li>
            <li>
              <a className="hover:underline block" href="https://anydice.com" onClick={() => setIsMenuOpen(false)}>
                AnyDice <ExternalWebsite />
              </a>
            </li>
            <li>
              <a className="hover:underline block" href="https://anydice.com/docs" onClick={() => setIsMenuOpen(false)}>
                AnyDice Documentation <ExternalWebsite />
              </a>
            </li>
          </ul>
        </nav>
        <div className="flex grow md:min-h-[400px]">
          <div className="flex flex-col md:flex-row w-full h-full items-stretch">
            <div className={`w-full md:w-1/2 p-4 ${borderColor} border`}>
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
            <div className={`w-full md:w-1/2 p-4 ${borderColor} border`}>
              <OutputPane distributions={output} />
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

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
