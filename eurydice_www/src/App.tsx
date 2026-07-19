import React, { useCallback, useEffect, useRef } from "react";

import { WorkerWrapper } from "./worker-wrapper";
import { Distribution } from "./util";
import OutputPane from "./components/OutputPane";
import EditorPane from "./components/EditorPane";
import Tutorial from "./components/Tutorial";
import Header from "./components/Header";
import { DarkModeSwitcher } from "./components/DarkModeSwitcher";
import EurydiceWorker from "./worker?worker";
import { Toaster } from "react-hot-toast";
import { Group, Panel, Separator } from "react-resizable-panels";

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
  const [isDesktopLayout, setIsDesktopLayout] = React.useState(() =>
    window.matchMedia("(min-width: 768px)").matches
  );
  const runLiveRef = useRef(true);
  const runningRef = useRef(false);

  useEffect(() => {
    const mediaQuery = window.matchMedia("(min-width: 768px)");
    const updateLayout = () => setIsDesktopLayout(mediaQuery.matches);

    mediaQuery.addEventListener("change", updateLayout);
    return () => mediaQuery.removeEventListener("change", updateLayout);
  }, []);

  function setRunLive(val: boolean) {
    runLiveRef.current = val;
    setRunLiveInner(val);
    if (val) {
      localStorage.removeItem("eurydice0_run_live");
      run(editorText);
    } else {
      localStorage.setItem("eurydice0_run_live", "false");
    }
  }

  const attachOnMessage = useCallback((worker: WorkerWrapper) => {
    worker.setOnmessage((event: MessageEvent<EurydiceMessage>) => {
      if (event.data.Err !== undefined) {
        runningRef.current = false;
        setRunning(false);
        setError(event.data.Err);
      } else if (event.data.Ok !== undefined) {
        runningRef.current = false;
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
        const evt = event.data.Print as [string, string];
        setPrintOutputs((printOutputs) => [...printOutputs, evt]);
      }
    });
  }, []);

  const run = useCallback((val: string) => {
    if (runningRef.current) {
      worker.terminate();
      worker = new WorkerWrapper(new EurydiceWorker());
      attachOnMessage(worker);
    }
    runningRef.current = true;
    setRunning(true);
    setPrintOutputs([]);
    setError(null);
    worker.postMessage(val);
  }, [attachOnMessage]);

  useEffect(() => {
    attachOnMessage(worker);
  }, [attachOnMessage]);

  useEffect(() => {
    // Load the saved state from local storage
    const savedRunLive = localStorage.getItem("eurydice0_run_live") !== "false";
    if (!savedRunLive) {
      runLiveRef.current = false;
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
  }, [run]);

  const onChange = useCallback((val: string) => {
    setEditorText(val);
    localStorage.setItem("eurydice0_editor_program", val);
    if (runLiveRef.current) {
      run(val);
    }
  }, [run]);

  const tutorial = showTutorial ? (
    <Tutorial
      setEditorText={onChange}
      closeTutorial={() => setShowTutorial(false)}
    />
  ) : null;

  const editorPane = (
    <div className="h-full p-4">
      {tutorial}
      <EditorPane
        editorText={editorText}
        onChange={onChange}
        runLive={runLive}
        setRunLive={setRunLive}
        running={running}
        run={() => run(editorText)}
        error={error}
        printOutputs={printOutputs}
      />
    </div>
  );

  const outputPane = (
    <div className="output-pane h-full p-4">
      <OutputPane distributions={output} />
    </div>
  );

  return (
    <>
      <div><Toaster /></div>
      <div className="flex min-h-screen flex-col md:h-dvh md:min-h-0 md:overflow-hidden">
        <Header
          showTutorial={true}
          onTutorialClick={() => setShowTutorial(true)}
        />
        <div className="flex grow md:min-h-0">
          {isDesktopLayout ? (
            <Group
              className="w-full"
              orientation="horizontal"
              id="editor-results"
            >
              <Panel id="editor" defaultSize="50%" minSize="25%">
                {editorPane}
              </Panel>
              <Separator
                className="split-pane-separator"
                aria-label="Resize editor and results panes"
              />
              <Panel id="results" defaultSize="50%" minSize="25%">
                {outputPane}
              </Panel>
            </Group>
          ) : (
            <div className="flex w-full flex-col items-stretch">
              <div className="w-full border-b">{editorPane}</div>
              <div className="w-full">{outputPane}</div>
            </div>
          )}
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
