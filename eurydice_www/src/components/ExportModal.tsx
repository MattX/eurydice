import React from "react";
import toast from "react-hot-toast";
import {
  generateSpreadsheetCSV,
  generateAnyDiceFormatCSV,
  downloadCSV,
} from "../utils/csvExport";
import { NamedDistribution, isScalarDistribution } from "../util";

interface ExportModalProps {
  outputs: NamedDistribution[];
  isOpen: boolean;
  onClose: () => void;
}

export default function ExportModal({
  outputs,
  isOpen,
  onClose,
}: ExportModalProps) {
  const [csvFormat, setCsvFormat] = React.useState<"spreadsheet" | "anydice">(
    "spreadsheet"
  );
  const dialogRef = React.useRef<HTMLDialogElement>(null);
  // AnyDice format is numeric-only, so enum and tuple outputs are dropped there.
  const omittedFromAnyDice = outputs.filter(([, distribution]) =>
    !isScalarDistribution(distribution) ||
    distribution.fields[0].kind === "enum"
  ).length;

  const { csvContent, csvFilename } = React.useMemo(() => {
    if (outputs.length === 0) {
      return { csvContent: "", csvFilename: "" };
    }
    return {
      csvContent:
        csvFormat === "spreadsheet"
          ? generateSpreadsheetCSV(outputs)
          : generateAnyDiceFormatCSV(outputs),
      csvFilename:
        csvFormat === "spreadsheet"
          ? "distributions.csv"
          : "distributions_anydice.csv",
    };
  }, [csvFormat, outputs]);

  // Handle modal open/close
  React.useEffect(() => {
    const dialog = dialogRef.current;
    if (isOpen && !dialog?.open) {
      dialog?.showModal();
    } else if (!isOpen && dialog?.open) {
      dialog.close();
    }
  }, [isOpen]);

  const handleCopyToClipboard = React.useCallback(async () => {
    try {
      await navigator.clipboard.writeText(csvContent);
      toast.success("CSV copied to clipboard!");
      onClose();
    } catch (err) {
      console.error("Failed to copy to clipboard:", err);
      toast.error("Failed to copy to clipboard");
    }
  }, [csvContent, onClose]);

  const handleDownload = React.useCallback(() => {
    downloadCSV(csvContent, csvFilename);
    onClose();
  }, [csvContent, csvFilename, onClose]);

  const handleDialogClick = (e: React.MouseEvent<HTMLDialogElement>) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <dialog
      ref={dialogRef}
      className="top-1/2 left-1/2 w-full max-w-2xl -translate-x-1/2 -translate-y-1/2 rounded-xl p-6 shadow-2xl backdrop:bg-black/40 backdrop:backdrop-blur-sm"
      style={{
        background: "var(--surface)",
        color: "var(--text)",
        border: "1px solid var(--border-strong)",
      }}
      onClose={onClose}
      onClick={handleDialogClick}
    >
      <h2 className="mb-3 text-lg font-semibold">Export distributions</h2>
      <div className="segmented mb-4" role="group" aria-label="Export format">
        <button
          aria-pressed={csvFormat === "spreadsheet"}
          onClick={() => setCsvFormat("spreadsheet")}
        >
          Spreadsheet
        </button>
        <button
          aria-pressed={csvFormat === "anydice"}
          onClick={() => setCsvFormat("anydice")}
        >
          AnyDice format
        </button>
      </div>
      {csvFormat === "anydice" && omittedFromAnyDice > 0 && (
        <p className="mb-3 text-sm text-[var(--text-muted)]">
          AnyDice format supports numeric outputs only. {omittedFromAnyDice}{" "}
          non-numeric {omittedFromAnyDice === 1 ? "output is" : "outputs are"}{" "}
          omitted.
        </p>
      )}
      <textarea
        value={csvContent}
        readOnly
        className="field h-48 w-full resize-none font-mono"
      />
      <div className="mt-4 flex gap-2">
        <button
          onClick={handleCopyToClipboard}
          className="btn btn-primary"
          disabled={csvContent.length === 0}
        >
          Copy to clipboard
        </button>
        <button
          onClick={handleDownload}
          className="btn btn-secondary"
          disabled={csvContent.length === 0}
        >
          Download
        </button>
        <button onClick={onClose} className="btn btn-ghost ml-auto">
          Close
        </button>
      </div>
    </dialog>
  );
}
