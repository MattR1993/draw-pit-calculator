"""Flow path analysis for N4ce DTMs.

Pipeline: DTM triangles/points -> regular grid -> slope + D8 flow ->
flow accumulation -> PNG (slope colour chart, slope arrows, flow paths).
Requires numpy and matplotlib (scipy optional, gives better interpolation).
"""
import numpy as np

# D8 neighbour offsets (dr, dc) and distance factors
D8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def grid_from_points(points, cell_size):
    """Interpolate an (N,3) array of x,y,z DTM points to a regular grid.
    Returns (xs, ys, z) with NaN outside the DTM."""
    pts = np.asarray(points, dtype=float)
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    xs = np.arange(x0, pts[:, 0].max() + cell_size, cell_size)
    ys = np.arange(y0, pts[:, 1].max() + cell_size, cell_size)
    gx, gy = np.meshgrid(xs, ys)
    try:
        from scipy.interpolate import LinearNDInterpolator
        z = LinearNDInterpolator(pts[:, :2], pts[:, 2])(gx, gy)
    except ImportError:  # inverse-distance fallback
        z = np.full(gx.shape, np.nan)
        for i in range(gx.shape[0]):
            d = np.hypot(pts[:, 0][None, :] - gx[i][:, None],
                         pts[:, 1][None, :] - gy[i][:, None]) + 1e-9
            w = 1 / d ** 2
            z[i] = (w * pts[:, 2]).sum(1) / w.sum(1)
            z[i][d.min(1) > cell_size * 3] = np.nan
    return xs, ys, z


def slope_and_direction(z, cell_size):
    """Return slope (%) and downhill vector components (dx, dy) per cell."""
    gy, gx = np.gradient(z, cell_size)  # row index increases with y
    slope = np.hypot(gx, gy) * 100.0
    mag = np.hypot(gx, gy)
    mag[mag == 0] = 1
    return slope, -gx / mag, -gy / mag


def d8_flow(z, cell_size):
    """Steepest-descent receiver index per cell (-1 = sink/edge)."""
    rows, cols = z.shape
    recv = np.full((rows, cols), -1, dtype=np.int64)
    best = np.zeros((rows, cols))
    for dr, dc in D8:
        dist = cell_size * (np.hypot(dr, dc))
        zr = np.full_like(z, np.nan)
        rs = slice(max(dr, 0), rows + min(dr, 0))
        cs = slice(max(dc, 0), cols + min(dc, 0))
        rd = slice(max(-dr, 0), rows + min(-dr, 0))
        cd = slice(max(-dc, 0), cols + min(-dc, 0))
        zr[rd, cd] = z[rs, cs]
        drop = (z - zr) / dist
        better = np.isfinite(drop) & (drop > best)
        best[better] = drop[better]
        rr, cc = np.indices(z.shape)
        recv[better] = ((rr + dr) * cols + (cc + dc))[better]
    recv[~np.isfinite(z)] = -1
    return recv


def flow_accumulation(z, recv):
    """Number of upstream cells draining through each cell."""
    acc = np.where(np.isfinite(z), 1.0, 0.0).ravel()
    zf = np.nan_to_num(z, nan=-np.inf).ravel()
    rf = recv.ravel()
    for i in np.argsort(-zf):  # high to low
        if rf[i] >= 0:
            acc[rf[i]] += acc[i]
    return acc.reshape(z.shape)


def analyse(points, cell_size=1.0):
    xs, ys, z = grid_from_points(points, cell_size)
    slope, ux, uy = slope_and_direction(z, cell_size)
    acc = flow_accumulation(z, d8_flow(z, cell_size))
    return dict(xs=xs, ys=ys, z=z, slope=slope, ux=ux, uy=uy, acc=acc)


def render(result, path, arrow_spacing=None, flow_threshold=None, cmap="turbo"):
    """Save a PNG: slope colour chart, slope arrows, flow paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs, ys = result["xs"], result["ys"]
    ext = [xs[0], xs[-1], ys[0], ys[-1]]
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(np.ma.masked_invalid(result["slope"]), origin="lower",
                   extent=ext, cmap=cmap)
    fig.colorbar(im, ax=ax, label="Slope (%)")
    step = arrow_spacing or max(1, int(max(len(xs), len(ys)) / 30))
    gx, gy = np.meshgrid(xs, ys)
    s = (slice(None, None, step),) * 2
    ax.quiver(gx[s], gy[s], result["ux"][s], result["uy"][s], color="k",
              scale=40, width=0.002)
    acc = result["acc"]
    thr = flow_threshold or max(10, np.nanpercentile(acc[acc > 0], 95))
    paths = np.ma.masked_where(acc < thr, acc)
    ax.imshow(paths, origin="lower", extent=ext, cmap="Blues_r", alpha=0.9)
    ax.set_aspect("equal")
    ax.set_title("Flow path analysis (arrows = downhill, blue = flow paths)")
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
