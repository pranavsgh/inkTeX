import { useDropzone } from "react-dropzone";

// Drag-and-drop + click-to-upload component for selecting a page image. (Mutha)
export default function UploadZone({ onFileSelected }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { "image/*": [] },
    multiple: false,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        onFileSelected(acceptedFiles[0]);
      }
    },
  });

  return (
    <div
      {...getRootProps()}
      className={`cursor-pointer rounded-lg border-2 border-dashed px-6 py-16 text-center transition-colors ${
        isDragActive ? "border-blue-400 bg-blue-50" : "border-gray-300 hover:border-gray-400"
      }`}
    >
      <input {...getInputProps()} />
      <p className="text-gray-600">
        {isDragActive ? "Drop the image here…" : "Drag & drop a handwritten page, or click to select"}
      </p>
      <p className="mt-1 text-sm text-gray-400">PNG, JPG</p>
    </div>
  );
}
