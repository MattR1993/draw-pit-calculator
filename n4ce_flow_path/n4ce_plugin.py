"""N4ce Apps plugin entry point for flow path analysis.

NOTE: the N4ce scripting API is vendor-specific. All N4ce calls are isolated
in get_dtm_points(); adapt it to your N4ce version's Python API if names differ.
Run standalone with a CSV (x,y,z): python n4ce_plugin.py dtm.csv out.png [cell]
"""
import sys
import numpy as np
from flow_path_analysis import analyse, render


def get_dtm_points():
    """Return an (N,3) array of the selected DTM's vertices from N4ce."""
    import n4ce  # provided by the N4ce Apps Python host (assumed module name)
    dtm = n4ce.get_selected_dtm()  # assumed API: adapt to your N4ce version
    return np.array([[v.x, v.y, v.z] for v in dtm.vertices], dtype=float)


def run(out_png="flow_paths.png", cell_size=1.0, points=None):
    pts = get_dtm_points() if points is None else points
    return render(analyse(pts, cell_size), out_png)


if __name__ == "__main__":
    data = np.loadtxt(sys.argv[1], delimiter=",")
    print(run(sys.argv[2] if len(sys.argv) > 2 else "flow_paths.png",
              float(sys.argv[3]) if len(sys.argv) > 3 else 1.0, data))
