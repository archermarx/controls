import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from lib.surrogate import Surrogate

def rosenbrock(x):
    x1, x2 = x
    a, b = 1, 100
    return (a - x1)**2 + b * (x2 - x1**2)**2

def himmelblau(x):
    x1, x2 = x
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

def booth(x):
    x1, x2 = x
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2

def bulkin6(x):
    x1, x2 = x
    return 100 * np.sqrt(np.abs(x2 - 0.01 * x1**2)) + 0.01 * np.abs(x1 + 10)

test_fns = {
    "rosenbrock": {
        "f": rosenbrock,
        "lb": [-2.0, -1.0],
        "ub": [2.0, 3.0],
        "minima": [(1,1)],
    },
    "himmelblau": {
        "f": himmelblau,
        "lb": [-4.0, -4.0],
        "ub": [4.0, 4.0],
        "minima": [
            (3.0, 2.0),
            (-2.805118, 3.131312),
            (-3.779310, -3.283196),
            (3.584428, -1.848126)
        ],
    },
    "booth": {
        "f": booth,
        "lb": [-10.0, -10.0],
        "ub": [10.0, 10.0],
        "minima": [(1,3)],
    },
    "bulkin6": {
        "f": bulkin6,
        "lb": [-15, -3],
        "ub": [-5, 3],
        "minima": [(-10, 1)],
    }
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", choices=["KRG", "KPLS"], default="KRG")
parser.add_argument("--function", "-f", choices=list(test_fns.keys()))
parser.add_argument("--num-iters", "-n", type=int, default=25)
parser.add_argument("--noise-std", "-s", type=float, default=0.0)

args = parser.parse_args()

fn = args.function
objective=test_fns[fn]["f"]
lb, ub = test_fns[fn]["lb"], test_fns[fn]["ub"]
num_iters = args.num_iters

surr = Surrogate(
    bounds=(lb, ub),
    model_type=args.model,
    kernel="squar_exp" if args.model =="KPLS" else "matern32"
)

d = surr.dim

def plot_surrogate(surr, obj):
    fig, axs = plt.subplots(1, 2, layout='constrained', figsize=(10,5))

    m = 100
    x1 = np.linspace(lb[0], ub[0], m)
    x2 = np.linspace(lb[1], ub[1], m)
    X1, X2 = np.meshgrid(x1, x2)

    X = np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1)))
    f = np.array([obj(x) for x in X]).reshape(m, m)

    vmin=np.min(f)
    vmax=np.max(f)
    cmap = 'viridis'
    locator = MaxNLocator(nbins=20, symmetric=False)
    levels = locator.tick_values(vmin, vmax)
    axs[0].contourf(X1, X2, f, levels=levels, cmap=cmap)

    f_surr = np.array([surr(x) for x in X]).reshape(m, m)
    axs[1].contourf(X1, X2, f_surr, levels=levels, cmap=cmap)
    axs[1].scatter(surr.X[:, 0], surr.X[:, 1], color = 'red')

    final_prediction, _ = surr.optimize(acquisition="mean", tol=0)

    for (x0, y0) in test_fns[fn]["minima"]:
        for ax in axs:
            ax.scatter([x0], [y0], 256, color='yellow', linewidth=1, edgecolor='black', marker='*')

    axs[1].scatter([final_prediction[0]], [final_prediction[1]], 36, color='cyan', linewidth=1, edgecolor='black')

    fig.savefig(f"{fn}_{surr.model_type.casefold()}_{num_iters:03d}.png")
    plt.close(fig)

start_points = 5
start_x = surr.sample_in_bounds(start_points)
start_y = np.array([objective(x) for x in start_x]) 

surr.update(start_x, start_y)

for i in range(num_iters):
    x, _ = surr.optimize(acquisition='ei', tol=1e-2)
    y = objective(x) + np.random.normal() * args.noise_std
    surr.update(x, y)
    print(f"Iter: {i+1}: minimum: {y}")

    if (i+1)%5 == 0:
        plot_surrogate(surr, objective)

plot_surrogate(surr, objective)






