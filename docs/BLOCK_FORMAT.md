# Block Format

The `Block` is the intermediate representation passed between layout
detection, recognition, and LaTeX assembly. It is defined as a Pydantic model
in [`backend/app/models/schemas.py`](../backend/app/models/schemas.py).

## Schema

| field | type | description |
|---|---|---|
| `type` | `"text" \| "math"` | Which recognizer produced/should produce `content` |
| `bbox` | `[x1, y1, x2, y2]` | Bounding box in pixel coordinates of the source image |
| `content` | `string` | Recognized text, or recognized/assembled LaTeX for math blocks |
| `confidence` | `float` | Model confidence in `[0.0, 1.0]` |
| `order` | `int` | Reading order of the block on the page, starting at 0 |

## Example

A page with one line of prose followed by two inline equations:

```json
[
  {
    "type": "text",
    "bbox": [12.0, 8.0, 410.0, 55.0],
    "content": "Let x and y satisfy the relation",
    "confidence": 0.97,
    "order": 0
  },
  {
    "type": "math",
    "bbox": [15.0, 60.0, 180.0, 105.0],
    "content": "x^2 + y^2 = z^2",
    "confidence": 0.93,
    "order": 1
  },
  {
    "type": "math",
    "bbox": [200.0, 60.0, 340.0, 105.0],
    "content": "\\frac{dx}{dt} = -y",
    "confidence": 0.89,
    "order": 2
  }
]
```

`latex_assembler.py` consumes blocks in ascending `order`, wrapping `math`
blocks in `$...$` and concatenating `text` blocks as plain prose, to build the
final `.tex` document.
