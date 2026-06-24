# API Contract

All three endpoints accept a multipart file upload under the field name `file`
and return JSON. Schemas referenced below are defined in
[`backend/app/models/schemas.py`](../backend/app/models/schemas.py).

## POST /math

Recognizes the LaTeX for a single cropped math block image. (Pranav)

**Request** — `multipart/form-data`

| field | type | description |
|---|---|---|
| `file` | file | Cropped image containing a single handwritten math expression |

**Response** — `200 OK`, `MathResponse`

```json
{
  "latex": "x^2 + y^2 = z^2",
  "confidence": 0.94
}
```

## POST /analyze

Runs layout detection on a full handwritten page and returns ordered blocks. (Mutha)

**Request** — `multipart/form-data`

| field | type | description |
|---|---|---|
| `file` | file | Full page image (text + math mixed) |

**Response** — `200 OK`, `AnalyzeResponse`

```json
{
  "blocks": [
    {
      "type": "text",
      "bbox": { "x1": 10.0, "y1": 12.0, "x2": 400.0, "y2": 60.0 },
      "content": "Consider the equation",
      "confidence": 0.98,
      "order": 0
    },
    {
      "type": "math",
      "bbox": { "x1": 10.0, "y1": 65.0, "x2": 200.0, "y2": 110.0 },
      "content": "x^2 + y^2 = z^2",
      "confidence": 0.94,
      "order": 1
    }
  ],
  "image_width": 1024,
  "image_height": 1448
}
```

## POST /convert

Master endpoint: analyze the page, route each block to the math or text model,
assemble the results into a `.tex` document, and compile it to a PDF. (Shared)

**Request** — `multipart/form-data`

| field | type | description |
|---|---|---|
| `file` | file | Full page image to convert end-to-end |

**Response** — `200 OK`, `ConvertResponse`

```json
{
  "tex_source": "\\documentclass{article}\n\\begin{document}\n...\n\\end{document}",
  "pdf_url": "/static/results/abc123.pdf"
}
```

`pdf_url` is `null` if PDF compilation failed; `tex_source` is still returned
in that case so the caller can inspect or fix the LaTeX manually.
