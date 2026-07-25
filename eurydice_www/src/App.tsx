import React, { useCallback, useEffect, useRef } from "react";

import { WorkerWrapper } from "./worker-wrapper";
import { NamedDistribution, isNamedScalarDistribution } from "./util";
import { WireDistribution, normalizeDistribution } from "./utils/tupleData";
import OutputPane from "./components/OutputPane";
import ExportModal from "./components/ExportModal";
import EditorPane from "./components/EditorPane";
import Tutorial from "./components/Tutorial";
import Header from "./components/Header";
import { DarkModeSwitcher } from "./components/DarkModeSwitcher";
import EurydiceWorker from "./worker?worker";
import { Toaster } from "react-hot-toast";
import { numericOutcomeRange } from "./utils/chartData";
import { Group, Panel, Separator } from "react-resizable-panels";

export default function App() {
  return (
    <DarkModeSwitcher>
      <AppInner />
    </DarkModeSwitcher>
  );
}

function AppInner() {
  const [editorText, setEditorText] = React.useState(() => {
    const hash = window.location.hash;
    return hash.startsWith("#p=")
      ? decodeURIComponent(hash.slice(3))
      : localStorage.getItem("eurydice0_editor_program") || "output 1d6 + 2";
  });
  const [output, setOutput] = React.useState<NamedDistribution[]>([]);
  const [error, setError] = React.useState<EurydiceError | null>(null);
  const [runLive, setRunLiveInner] = React.useState(
    () => localStorage.getItem("eurydice0_run_live") !== "false"
  );
  const [running, setRunning] = React.useState(false);
  const [printOutputs, setPrintOutputs] = React.useState<[string, string][]>(
    [],
  );
  const [showTutorial, setShowTutorial] = React.useState(false);
  const [showExportModal, setShowExportModal] = React.useState(false);
  const [isDesktopLayout, setIsDesktopLayout] = React.useState(() =>
    window.matchMedia("(min-width: 768px)").matches
  );
  const runLiveRef = useRef(runLive);
  const initialEditorTextRef = useRef(editorText);
  const runningRef = useRef(false);
  const workerRef = useRef<WorkerWrapper | null>(null);

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
    worker.setOnMessage((event: MessageEvent<EurydiceMessage>) => {
      if ("Err" in event.data) {
        runningRef.current = false;
        setRunning(false);
        setError(event.data.Err);
      } else if ("Ok" in event.data) {
        runningRef.current = false;
        setRunning(false);
        setError(null);
        const distributions: NamedDistribution[] = event.data.Ok.map(
          ([name, distribution]) => [name, normalizeDistribution(distribution)]
        );
        const chartData = distributions.filter(isNamedScalarDistribution);

        // Categorical outcomes use enum member ordinals, not a numeric axis.
        const range = numericOutcomeRange(chartData);
        if (range !== null && range >= 5000) {
          setOutput(
            distributions.filter(
              ([, distribution]) => distribution.fields.length > 1
            )
          );
          setError({
            message: `Range of outcomes (${range}) is too large to display. Maximum range is 5000.`,
            from: 0,
            to: 0,
          });
        } else {
          setOutput(distributions);
        }
      } else if ("Print" in event.data) {
        const printOutput = event.data.Print;
        setPrintOutputs((printOutputs) => [...printOutputs, printOutput]);
      }
    });
  }, []);

  const run = useCallback((val: string) => {
    let worker = workerRef.current;
    if (worker === null) {
      worker = new WorkerWrapper(new EurydiceWorker());
      attachOnMessage(worker);
      workerRef.current = worker;
    }
    if (runningRef.current) {
      worker.terminate();
      worker = new WorkerWrapper(new EurydiceWorker());
      attachOnMessage(worker);
      workerRef.current = worker;
    }
    runningRef.current = true;
    setRunning(true);
    setPrintOutputs([]);
    setError(null);
    worker.postMessage(val);
  }, [attachOnMessage]);

  useEffect(() => {
    const worker = new WorkerWrapper(new EurydiceWorker());
    attachOnMessage(worker);
    workerRef.current = worker;

    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
      runningRef.current = false;
    };
  }, [attachOnMessage]);

  useEffect(() => {
    if (runLiveRef.current) {
      run(initialEditorTextRef.current);
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

  // Export lives in the shared toolbar rather than an output section so it's
  // clearly a global action over every output, and never wraps onto its own
  // line inside the results pane.
  const canExport = output.length > 0;
  const exportButton = (
    <button
      className="btn btn-secondary"
      disabled={!canExport}
      onClick={() => setShowExportModal(true)}
    >
      Export
    </button>
  );

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
        exportButton={exportButton}
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
      <div>
        <Toaster />
      </div>
      <div className="flex min-h-screen flex-col md:h-dvh md:min-h-0 md:overflow-hidden">
        <Header onTutorialClick={() => setShowTutorial(true)} />
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

      <ExportModal
        outputs={output}
        isOpen={showExportModal}
        onClose={() => setShowExportModal(false)}
      />
    </>
  );
}

type EurydiceMessage =
  | { Ok: [string, WireDistribution][] }
  | { Err: EurydiceError }
  | { Print: [string, string] };

interface EurydiceError {
  message: string;
  from: number;
  to: number;
}
