import axios from "axios";

// Axios instance pointed at the backend API; base URL is injected via Vite env vars.
const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
});

// Posts `file` under the `file` field as multipart/form-data. Content-Type (with the
// multipart boundary) is left for axios/the browser to set automatically.
function postFile(path, file) {
  const formData = new FormData();
  formData.append("file", file);
  return client.post(path, formData);
}

/**
 * Upload a full handwritten page image for layout analysis.
 * @param {File} file
 * @returns {Promise<import("axios").AxiosResponse>}
 */
export function uploadPage(file) {
  return postFile("/analyze", file);
}

/**
 * Upload a single cropped math block image for LaTeX recognition.
 * @param {File} file
 * @returns {Promise<import("axios").AxiosResponse>}
 */
export function getMath(file) {
  return postFile("/math", file);
}

/**
 * Run the full convert pipeline on a page image, returning tex source and PDF URL.
 * @param {File} file
 * @returns {Promise<import("axios").AxiosResponse>}
 */
export function convert(file) {
  return postFile("/convert", file);
}

export default client;
