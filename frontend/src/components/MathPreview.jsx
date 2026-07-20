import { useMemo } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

// Renders a LaTeX string as live-rendered math using KaTeX. (Pranav)
export default function MathPreview({ latex }) {
  const { html, error } = useMemo(() => {
    if (!latex) return { html: "", error: null };
    try {
      const rendered = katex.renderToString(latex, {
        throwOnError: true,
        displayMode: true,
      });
      return { html: rendered, error: null };
    } catch (err) {
      return { html: "", error: err.message };
    }
  }, [latex]);

  if (!latex) {
    return <div className="text-sm italic text-gray-400">No expression to preview</div>;
  }

  if (error) {
    return (
      <div className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-600">
        Couldn't render LaTeX: {error}
      </div>
    );
  }

  return <div className="overflow-x-auto py-2" dangerouslySetInnerHTML={{ __html: html }} />;
}
