import { useEffect, useRef } from "react";

// Canvas overlay drawing detected text/math region bounding boxes over the uploaded image. (Mutha)
export default function BoundingBoxOverlay({ imageUrl, blocks }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!imageUrl) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const image = new Image();

    image.onload = () => {
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      ctx.drawImage(image, 0, 0);

      for (const block of blocks || []) {
        const { x1, y1, x2, y2 } = block.bbox;
        ctx.strokeStyle = block.type === "math" ? "#a855f7" : "#3b82f6";
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }
    };
    image.src = imageUrl;
  }, [imageUrl, blocks]);

  return <canvas ref={canvasRef} className="h-auto max-w-full rounded border border-gray-200" />;
}
