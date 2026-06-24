# inkTeX

Deep learning pipeline that converts handwritten math and text into compiled LaTeX documents.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Node](https://img.shields.io/badge/node-20%2B-green)
![Status](https://img.shields.io/badge/status-scaffold-lightgrey)

## Overview

inkTeX takes a photo or scan of a handwritten page — equations, prose, or a mix
of both — and produces a compiled PDF backed by real LaTeX source. A layout
detector segments the page into blocks, math blocks are routed through a
custom encoder-decoder model trained on im2latex-100k, text blocks are routed
through an OCR model, and the results are assembled into a `.tex` document
and compiled with `pdflatex`.

## Architecture

> Diagram placeholder — page image → layout detection (YOLOv8) → block
> routing (math vs. text) → math recognition (CNN/ViT encoder + Transformer
> decoder) / text OCR (TrOCR) → LaTeX assembly → `pdflatex` → PDF.

## Setup

### Backend

```bash
cd backend
pip install -e .
cp ../.env.example ../.env
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Project Structure

See [docs/API_CONTRACT.md](docs/API_CONTRACT.md) and
[docs/BLOCK_FORMAT.md](docs/BLOCK_FORMAT.md) for the API and data contracts
shared between backend and frontend.

## Team

- **Pranav** — math recognition pipeline (encoder/decoder model, training,
  math endpoint, PDF compilation)
- **Mutha** — layout detection and OCR pipeline (YOLOv8 segmentation, TrOCR,
  upload/result UI)

## License

[MIT](LICENSE)
