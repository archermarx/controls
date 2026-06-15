import argparse
import json
import os
import pickle
import re

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
import imageio.v2 as imageio

from lib import controls as controls
from lib import labview as labview
from lib import surrogate as surrogate

def _ax_plot_surrogate_1d(ax, xs, ys, metadata, iter):
    lb = metadata["lower_bound"]
    ub = metadata["upper_bound"]

    surr = surrogate.Surrogate(bounds=(lb, ub), kernel="matern52")
    surr.X = xs
    surr.Y = ys
    surr._fit()

    x = np.linspace(lb, ub, 101)
    x_scaled = surr._scale(x)
    assert surr.model is not None
    surr_val = surr.model.predict_values(x_scaled).squeeze(-1)
    surr_std = np.sqrt(np.maximum(surr.model.predict_variances(x_scaled), 0.0)).squeeze(-1)

    ax.fill_between(x, surr_val-2*surr_std, surr_val+2*surr_std, alpha=0.3, color='tab:blue', zorder=1, linewidth=0)
    ax.plot(x, surr_val, color='tab:blue', zorder=2, linewidth=2)
    ax.scatter(xs[:-1], ys[:-1], s=49, color='red', linewidth=1, edgecolor='black', zorder=3)
    ax.scatter(xs[-1], ys[-1], s=200, color='yellow', marker='*', linewidth=1, edgecolor='black', zorder=4)
    ax.set(xlim=(lb, ub), title=f"Iteration {iter}", xlabel=metadata["variable_name"], ylabel=metadata["metric_name"])
    ax.set_ylim(metadata.get("ylims", (None, None)))

def _ax_plot_surrogate_2d(ax, xs, ys, metadata, iter):
    lb = metadata["lower_bound"]
    ub = metadata["upper_bound"]
    
    surr = surrogate.Surrogate(bounds=(lb, ub))
    surr.X = xs
    surr.Y = ys
    surr._fit()
    assert surr.model is not None

    vmin, vmax = metadata["ylims"]
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    m = 25
    grid_x1 = np.linspace(lb[0], ub[0], m)
    grid_x2 = np.linspace(lb[1], ub[1], m)
    X1, X2 = np.meshgrid(grid_x1, grid_x2)
    surr_pts = np.concat([X1.reshape(-1, 1), X2.reshape(-1, 1)], axis=1)
    assert surr_pts.shape==(m*m, 2)

    surr_pts_scaled = surr._scale(surr_pts)
    surr_vals = surr.model.predict_values(surr_pts_scaled)

    cmap = 'viridis'
    locator = MaxNLocator(nbins=20, symmetric=False)
    levels = locator.tick_values(vmin, vmax)

    cf = ax.contourf(X1, X2, surr_vals.reshape(m, m), levels=levels, cmap=cmap, norm=norm)
    xs = np.array(xs)
    ax.scatter(xs[:, 0], xs[:, 1], s=49, c = ys, cmap=cmap, norm=norm, marker='s', linewidth=1, edgecolor='white')
    
    def pad_limits(lim, pad_factor=0.025):
        pad = pad_factor * (lim[1] - lim[0])
        return lim[0]-pad, lim[1]+pad

    ax.set(
        xlim=pad_limits((lb[0], ub[0])), ylim=pad_limits((lb[1], ub[1])),
        xlabel = metadata["variable_name"][0],
        ylabel = metadata["variable_name"][1],
        title = f"Iteration {iter}",
        aspect = 'auto'
    )
    return cf

def plot_surrogate(xs, ys, metadata, iter, output: Path = Path(".")):
    fig, ax = plt.subplots(1,1)
    dim = len(xs[0])
    if dim == 1:
        _ax_plot_surrogate_1d(ax, xs, ys, metadata, iter)
    else:
        cf = _ax_plot_surrogate_2d(ax, xs, ys, metadata, iter)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(cf, cax=cax, extend='both', label=metadata["metric_name"])
        cb.mappable.set_clim(*metadata["ylims"])

    im_path = output / f"surrogate_{iter:03d}.png"
    fig.savefig(im_path , dpi=200)
    plt.close(fig)
    return im_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path, help="directory containing surrogate optimization records")
    args = parser.parse_args()

    files = os.listdir(args.dir)
    regex = re.compile(r"surrogate_\d+.pkl")
    history_files = [args.dir / f for f in files if regex.match(f)]

    plot_dir = args.dir / "plots"
    os.makedirs(plot_dir, exist_ok=True)

    metadata_file = args.dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "rb") as fd:
            metadata = json.load(fd)
    else:
        metadata = {}

    xs = []
    ys = []
    for file in history_files:
        with open(file, "rb") as fd:
            contents = pickle.load(fd)
        xs.append(contents["control_vector"])
        ys.append(contents["z_actual"])
        print(f"{contents["setpoint"]=}")

    dim = len(xs[1])

    if dim == 1 or dim == 2: 
        # Plot each frame as a png image
        image_paths = []
        for i in range(len(xs)):
            if i > dim:
                im_path = plot_surrogate(xs[:i], ys[:i], iter=i, metadata=metadata, output=plot_dir)
                image_paths.append(im_path)
                print(str(im_path))

    # Save animated gif
    images = [imageio.imread(f) for f in image_paths]
    durations = [1000] * len(images)    # milliseconds per frame
    durations[-1] = 2000
    imageio.mimsave(plot_dir / "optimization.gif", images, duration=durations, loop=0) #type:ignore

if __name__ == "__main__":
    main()