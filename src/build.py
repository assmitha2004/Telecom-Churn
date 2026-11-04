"""Inject the trained model, metrics and sample batch into the UI template -> index.html."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
body = (root / "src/template.html").read_text()
body = (body.replace("__MODEL__", (root / "out/model.json").read_text())
            .replace("__METRICS__", (root / "out/metrics.json").read_text())
            .replace("__SAMPLE__", json.dumps((root / "out/sample_batch.csv").read_text())))
page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style>\n'
        + body.replace("<div class=\"wrap\">", "</head>\n<body>\n<div class=\"wrap\">", 1)
        + "\n</body>\n</html>\n")
(root / "index.html").write_text(page)
print("Wrote index.html", len(page), "bytes")
