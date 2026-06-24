import axios from "axios";

// Axios instance pointed at the backend API; base URL is injected via Vite env vars.
const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
});

/**
 * Upload a full handwritten page image for layout analysis.
 * @param {File} file
 * @returns {Promise<import("axios").AxiosResponse>}
 */
export function uploadPage(file) {
  throw new Error("Not implemented");
}

/**
 * Upload a single cropped math block image for LaTeX recognition.
 * @param {File} file
 * @returns {Promise<import("axios").AxiosResponse>}
 */
export function getMath(file) {
  throw new Error("Not implemented");
}

/**
 * Run the full convert pipeline on a page image, returning tex source and PDF URL.
 * @param {File} file
 * @returns {Promise<import("axios").AxiosResponse>}
 */
export function convert(file) {
  throw new Error("Not implemented");
}

export default client;
