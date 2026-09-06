# Charts and source data

The final set has10 charts; the earlier set has7. A separate prefix-cache
addendum and the generated header banner are also included. Earlier data stays
labeled historical. Public copies use anonymous paths.

PNG files are ready to attach. SVG files are editable masters. The PDF and
self-contained HTML make the set easier to read on a phone. Chart data is in
`reports/final/chart_data.json`; prefix measurements also have a CSV export.

To redraw the10 final charts without loading a model:

```bash
python3 -m pip install matplotlib
python3 scripts/render-charts.py --out rendered-charts
```

The original plotting environment used Matplotlib3.6.3 and DejaVu Sans Mono.
A different Matplotlib/font version can change layout slightly. The image-gen
banner is an included bitmap; the chart script does not regenerate it.

AA data was retrieved from the authenticated official API. The chart keeps the
highest-scoring listed variant per unique model name and displays the leading models.
It keeps only the highest-scoring entry per model family across efforts and older versions.
