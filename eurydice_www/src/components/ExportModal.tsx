import React from "react";
import toast from "react-hot-toast";
import {
  generateValuesOnlyCSV,
  generateAnyDiceFormatCSV,
  downloadCSV,
  DistributionData,
} from "../utils/csvExport";

interface ExportModalProps {
  distributions: DistributionData[];
  isOpen: boolean;
  onClose: () => void;
}

export default function ExportModal({ distributions, isOpen, onClose }: ExportModalProps) {
  const [csvContent, setCsvContent] = React.useState("");
  const [csvFilename, setCsvFilename] = React.useState("");
  const [csvFormat, setCsvFormat] = React.useState<"values" | "anydice">("values");
  const dialogRef = React.useRef<HTMLDialogElement>(null);

  // Update CSV content when format or distributions change
  React.useEffect(() => {
    if (distributions.length > 0) {
      const generate = csvFormat === "values" ? generateValuesOnlyCSV : generateAnyDiceFormatCSV;
      const filename = csvFormat === "values" ? "distributions_values.csv" : "distributions_anydice.csv";
      const csv = generate(distributions);
      setCsvContent(csv);
      setCsvFilename(filename);
    }
  }, [csvFormat, distributions]);

  // Handle modal open/close
  React.useEffect(() => {
    if (isOpen) {
      dialogRef.current?.showModal();
    } else {
      dialogRef.current?.close();
    }
  }, [isOpen]);

  const handleCopyToClipboard = React.useCallback(async () => {
    try {
      await navigator.clipboard.writeText(csvContent);
      toast.success('CSV copied to clipboard!');
      onClose();
    } catch (err) {
      console.error('Failed to copy to clipboard:', err);
      toast.error('Failed to copy to clipboard');
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
      className="backdrop:backdrop-blur-sm rounded-lg p-6 max-w-2xl w-full max-h-96 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 dark:bg-slate-800 dark:text-slate-300 border-2 border-gray-500"
      onClose={onClose}
      onClick={handleDialogClick}
    >
      <div className="flex gap-2 mb-4">
        <label className="border-2 border-blue-500 hover:border-blue-700 py-1 px-2 rounded-sm align-middle">
          <input
            type="radio"
            name="csvFormat"
            checked={csvFormat === "values"}
            onChange={() => setCsvFormat("values")}
          />{" "}
          Value table
        </label>
        <label className="border-2 border-blue-500 hover:border-blue-700 py-1 px-2 rounded-sm align-middle">
          <input
            type="radio"
            name="csvFormat"
            checked={csvFormat === "anydice"}
            onChange={() => setCsvFormat("anydice")}
          />{" "}
          AnyDice format
        </label>
      </div>
      <textarea
        value={csvContent}
        readOnly
        className="w-full h-48 border border-gray-300 rounded p-2 font-mono text-sm resize-none"
      />
      <div className="flex gap-2 mt-4">
        <button
          onClick={handleCopyToClipboard}
          className="px-4 py-2 bg-blue-500 rounded hover:bg-blue-600"
        >
          Copy to clipboard
        </button>
        <button
          onClick={handleDownload}
          className="px-4 py-2 bg-blue-500 rounded hover:bg-blue-600"
        >
          Download
        </button>
        <button
          onClick={onClose}
          className="px-4 py-2 bg-gray-500 rounded hover:bg-gray-600 ml-auto"
        >
          Close
        </button>
      </div>
    </dialog>
  );
}