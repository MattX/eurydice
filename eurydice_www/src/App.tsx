import React, { useCallback, useEffect, useRef } from "react";

import { WorkerWrapper } from "./worker-wrapper";
import { NamedDistribution, isNamedScalarDistribution } from "./util";
import { normalizeDistribution } from "./utils/tupleData";
import OutputPane from "./components/OutputPane";
import ExportModal from "./components/ExportModal";
import EditorPane from "./components/EditorPane";
import Tutorial from "./components/Tutorial";
import Header from "./components/Header";
import { DarkModeSwitcher } from "./components/DarkModeSwitcher";
import EurydiceWorker from "./worker?worker";
import { Toaster } from "react-hot-toast";
import { numericChartOutcomeRange } from "./utils/chartData";
import { Group, Panel, Separator } from "react-resizable-panels";
import {
  currentSource,
  DiagnosticSource,
  EurydiceDiagnostic,
  RunReport,
} from "./diagnostics";
import { PrimitiveMetadata } from "./autocomplete";

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
  const [diagnostics, setDiagnostics] = React.useState<EurydiceDiagnostic[]>(
    [],
  );
  // The submission the diagnostics describe, kept whole so a suggested fix can
  // check that the editor still holds the text its offsets were measured in.
  const [diagnosticSource, setDiagnosticSource] =
    React.useState<DiagnosticSource | null>(null);
  const [primitives, setPrimitives] = React.useState<PrimitiveMetadata[]>([]);
  const [runLive, setRunLiveInner] = React.useState(
    () => localStorage.getItem("eurydice0_run_live") !== "false",
  );
  const [running, setRunning] = React.useState(false);
  const [printOutputs, setPrintOutputs] = React.useState<[string, string][]>(
    [],
  );
  const [showTutorial, setShowTutorial] = React.useState(false);
  const [showExportModal, setShowExportModal] = React.useState(false);
  const [isDesktopLayout, setIsDesktopLayout] = React.useState(
    () => window.matchMedia("(min-width: 768px)").matches,
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
      if ("Ready" in event.data) {
        // Every worker sends the same static metadata, and a run that is
        // interrupted starts a fresh one. Keeping the first array means the
        // editor is not reconfigured — closing any open completion — on each
        // restart.
        const { primitives } = event.data.Ready;
        setPrimitives((current) => (current.length > 0 ? current : primitives));
      } else if ("Report" in event.data) {
        runningRef.current = false;
        setRunning(false);
        const report = event.data.Report;
        const source = currentSource(report.diagnostics);
        const sourceId = source?.id ?? null;
        const nextDiagnostics: EurydiceDiagnostic[] = [
          ...report.diagnostics.entries,
        ];
        setDiagnosticSource(source);

        const distributions: NamedDistribution[] = report.outputs.map(
          ({ name, distribution }) => [
            name,
            normalizeDistribution(distribution),
          ],
        );
        const chartData = distributions.filter(isNamedScalarDistribution);

        // The numeric chart fills every integer between its endpoints. The
        // categorical chart only shows observed outcomes, so it needs no range
        // limit even when one of those outcomes is a large integer.
        const range = numericChartOutcomeRange(chartData);
        if (range !== null && range >= 5000) {
          setOutput(
            distributions.filter(
              ([, distribution]) => distribution.fields.length > 1,
            ),
          );
          nextDiagnostics.push(
            frontendDiagnostic(
              `Range of outcomes (${range}) is too large to display. Maximum range is 5000.`,
              sourceId,
            ),
          );
        } else if (
          !nextDiagnostics.some((diagnostic) => diagnostic.severity === "error")
        ) {
          setOutput(distributions);
        }
        setDiagnostics(nextDiagnostics);
      } else if ("InternalError" in event.data) {
        runningRef.current = false;
        setRunning(false);
        setDiagnosticSource(null);
        setDiagnostics([frontendDiagnostic(event.data.InternalError, null)]);
      } else if ("Print" in event.data) {
        const printOutput = event.data.Print;
        setPrintOutputs((printOutputs) => [...printOutputs, printOutput]);
      }
    });
  }, []);

  const run = useCallback(
    (val: string) => {
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
      setDiagnostics([]);
      worker.postMessage(val);
    },
    [attachOnMessage],
  );

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

  const onChange = useCallback(
    (val: string) => {
      setEditorText(val);
      localStorage.setItem("eurydice0_editor_program", val);
      if (runLiveRef.current) {
        run(val);
      }
    },
    [run],
  );

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
        diagnostics={diagnostics}
        diagnosticSource={diagnosticSource}
        primitives={primitives}
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
  | { Ready: { primitives: PrimitiveMetadata[] } }
  | { Report: RunReport }
  | { InternalError: string }
  | { Print: [string, string] };

function frontendDiagnostic(
  summary: string,
  sourceId: number | null,
): EurydiceDiagnostic {
  return {
    code: "frontend.display_error",
    severity: "error",
    summary,
    primary_label:
      sourceId === null
        ? null
        : {
            range: { source: sourceId, range: { start: 0, end: 0 } },
            message: "",
          },
    secondary_labels: [],
    help: null,
    fix: null,
    trace: [],
  };
}
