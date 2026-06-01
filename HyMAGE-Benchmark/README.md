# HyMAGE Benchmark Tools

This Flask app provides lightweight companion tools for HyMAGE-generated hypergraphs. It currently focuses on structural analysis and manual/AI-assisted annotation.

## Features

### Structural Metrics (P1-P8)

Evaluates eight fundamental structural patterns of real-world hypergraphs:

| # | Metric | Type | Comparison |
| --- | --- | --- | --- |
| P1 | Heavy-tailed degree distribution | Distribution | JS Divergence |
| P2 | Heavy-tailed hyperedge size distribution | Distribution | JS Divergence |
| P3 | Heavy-tailed intersection size distribution | Distribution | JS Divergence |
| P4 | Skewed singular value distribution | Trend | Pearson Correlation |
| P5 | Intersecting pairs | Trend | Pearson Correlation |
| P6 | Heavy-tailed group degree distribution | Distribution | JS Divergence |
| P7 | Heavy-tailed hypercoreness distribution | Distribution | JS Divergence |
| P8 | Power-law persistence | Distribution | JS Divergence |

### Hypergraph Annotation

- Supports collaboration, social, drug, protein, and custom network schemas.
- Lets users manage node profiles, hyperedges, labels, and custom fields.
- Provides optional OpenAI-compatible assistance for extracting structured metadata from text.
- Exports `profiles.json` and `hyperedges.txt`.

## Project Structure

```text
HyMAGE-Benchmark/
|-- app.py
|-- requirements.txt
|-- modules/
|   `-- structural_metrics.py
|-- templates/
|   |-- base.html
|   |-- index.html
|   |-- structural.html
|   `-- annotate.html
|-- static/
|   |-- css/style.css
|   `-- js/main.js
|-- uploads/       # auto-created
`-- annotations/   # auto-created
```

## Run

```bash
cd HyMAGE-Benchmark
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5001`.

## Dataset Format

Hyperedge files are plain text, one hyperedge per line:

```text
node1 node2 node3
node2 node5
```

Node profiles use JSON:

```json
{
  "node1": {"name": "Alice", "field": "AI"},
  "node2": {"name": "Bob", "field": "Systems"}
}
```
