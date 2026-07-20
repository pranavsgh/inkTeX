// Shows the ordered list of detected blocks with their type labels (text/math). (Pranav)
export default function BlockList({ blocks }) {
  if (!blocks || blocks.length === 0) {
    return <p className="text-sm text-gray-400">No blocks detected.</p>;
  }

  const ordered = blocks.slice().sort((a, b) => a.order - b.order);

  return (
    <ol className="space-y-2">
      {ordered.map((block) => (
        <li key={block.order} className="flex items-start gap-3 rounded border border-gray-200 px-3 py-2">
          <span
            className={`mt-0.5 rounded px-2 py-0.5 text-xs font-medium uppercase ${
              block.type === "math" ? "bg-purple-100 text-purple-700" : "bg-blue-100 text-blue-700"
            }`}
          >
            {block.type}
          </span>
          <span className="flex-1 text-sm text-gray-800">
            {block.content || <em className="text-gray-400">(empty)</em>}
          </span>
          <span className="text-xs text-gray-400">{Math.round(block.confidence * 100)}%</span>
        </li>
      ))}
    </ol>
  );
}
