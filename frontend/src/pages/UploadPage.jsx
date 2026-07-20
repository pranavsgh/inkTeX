import { useState } from "react";
import { useNavigate } from "react-router-dom";
import UploadZone from "../components/UploadZone.jsx";
import { convert, uploadPage as analyzePage } from "../api/client.js";

// Main upload interface: drop a page image and kick off the convert pipeline.
export default function UploadPage() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  async function handleFileSelected(file) {
    setIsProcessing(true);
    setError(null);

    const imageUrl = URL.createObjectURL(file);
    try {
      // Two separate uploads because /convert doesn't return per-block detections
      // and /analyze doesn't compile a PDF — each endpoint answers half of what
      // the result page needs.
      const [convertRes, analyzeRes] = await Promise.all([convert(file), analyzePage(file)]);

      navigate("/result", {
        state: {
          imageUrl,
          texSource: convertRes.data.tex_source,
          pdfUrl: convertRes.data.pdf_url,
          blocks: analyzeRes.data.blocks,
        },
      });
    } catch (err) {
      setError(err.message || "Something went wrong while converting your page.");
      setIsProcessing(false);
    }
  }

  return (
    <div className="mx-auto max-w-2xl px-6 py-12">
      <h1 className="mb-2 text-2xl font-semibold text-gray-900">inkTeX</h1>
      <p className="mb-8 text-gray-500">Upload a photo of a handwritten page and get back compiled LaTeX.</p>

      <UploadZone onFileSelected={handleFileSelected} />

      {isProcessing && <p className="mt-4 text-sm text-gray-500">Converting your page…</p>}
      {error && <p className="mt-4 text-sm text-red-600">{error}</p>}
    </div>
  );
}
