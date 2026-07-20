import { Link, useLocation } from "react-router-dom";
import BlockList from "../components/BlockList.jsx";
import BoundingBoxOverlay from "../components/BoundingBoxOverlay.jsx";
import client from "../api/client.js";

// Side-by-side view: original uploaded image vs. the rendered PDF output.
export default function ResultPage() {
  const { state } = useLocation();

  if (!state) {
    return (
      <div className="mx-auto max-w-2xl px-6 py-12">
        <p className="text-gray-500">
          No conversion result to show.{" "}
          <Link to="/" className="text-blue-600 underline">
            Upload a page
          </Link>{" "}
          first.
        </p>
      </div>
    );
  }

  const { imageUrl, pdfUrl, texSource, blocks } = state;
  // pdfUrl is a path relative to the backend (e.g. "/static/results/abc123.pdf"), not
  // the frontend's own origin, so it needs the same base URL the API client uses.
  const fullPdfUrl = pdfUrl ? `${client.defaults.baseURL}${pdfUrl}` : null;

  return (
    <div className="mx-auto max-w-5xl px-6 py-12">
      <Link to="/" className="text-sm text-blue-600 underline">
        ← Upload another page
      </Link>
      <h1 className="mb-6 mt-2 text-2xl font-semibold text-gray-900">Result</h1>

      <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
        <div>
          <h2 className="mb-2 text-sm font-medium uppercase text-gray-500">Original</h2>
          <BoundingBoxOverlay imageUrl={imageUrl} blocks={blocks} />
        </div>
        <div>
          <h2 className="mb-2 text-sm font-medium uppercase text-gray-500">Compiled PDF</h2>
          {fullPdfUrl ? (
            <iframe
              src={fullPdfUrl}
              title="Compiled PDF"
              className="h-[600px] w-full rounded border border-gray-200"
            />
          ) : (
            <div className="rounded border border-red-200 bg-red-50 p-4 text-sm text-red-600">
              PDF compilation failed. Raw LaTeX source:
              <pre className="mt-2 overflow-x-auto whitespace-pre-wrap text-xs">{texSource}</pre>
            </div>
          )}
        </div>
      </div>

      <div className="mt-8">
        <h2 className="mb-2 text-sm font-medium uppercase text-gray-500">Detected blocks</h2>
        <BlockList blocks={blocks} />
      </div>
    </div>
  );
}
