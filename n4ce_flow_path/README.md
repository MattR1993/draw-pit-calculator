# N4ce Flow Path Analysis

Python plugin that analyses a DTM and renders: slope colour chart (% gradient),
slope arrows (downhill direction) and D8 flow paths (accumulation).

Requires `numpy`, `matplotlib` (optional `scipy`). Standalone test:
`python n4ce_plugin.py dtm.csv out.png 1.0` (CSV of x,y,z).
Inside N4ce, call `n4ce_plugin.run(out_png, cell_size)`; adapt
`get_dtm_points()` to your N4ce version's API (the API names there are assumed).
