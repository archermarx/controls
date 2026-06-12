import argparse
import json
import os
import pickle
import re

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from lib import controls as controls
from lib import labview as labview
from lib import surrogate as surrogate

def _ax_plot_surrogate(ax, xs, ys, metadata, iter):
    lb = metadata["lower_bound"]
    ub = metadata["upper_bound"]
    bounds = [(lb, ub)]

    surr = surrogate.Surrogate(dim=1, bounds=bounds)
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

def plot_surrogate_iter(xs, ys, metadata, iter, output: Path = Path(".")):
    fig, ax = plt.subplots(1,1)
    _ax_plot_surrogate(ax, xs, ys, metadata, iter)
    im_path = output / f"surrogate_{iter:03d}.png"
    fig.savefig(im_path , dpi=200)
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
    trained = []
    for file in history_files:
        with open(file, "rb") as fd:
            contents = pickle.load(fd)
        xs.append(contents["control_vector"])
        ys.append(contents["z_actual"])
        trained.append(contents["surrogate_trained"])

    # Plot each frame as a png image
    image_paths = []
    for i in range(len(xs)):
        if trained[i]:
            im_path = plot_surrogate_iter(xs[:i], ys[:i], iter=i, metadata=metadata, output=plot_dir)
            image_paths.append(im_path)
            print(str(im_path))

    # Save animated gif
    images = [imageio.imread(f) for f in image_paths]
    durations = [1000] * len(images)    # milliseconds per frame
    durations[-1] = 2000
    imageio.mimsave(plot_dir / "optimization.gif", images, duration=durations, loop=0) #type:ignore

if __name__ == "__main__":
    main()