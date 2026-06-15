import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from lib.surrogate import Surrogate

lb = [-2.0, -1.0]
ub = [2.0, 3.0]

lb = [-4.0, -4.0]
ub = [4.0, 4.0]
surr = Surrogate(bounds=(lb, ub))
d = surr.dim
noise_std = 1e-3

def rosenbrock(x):
    x1, x2 = x
    a, b = 1, 100
    return np.log10((a - x1)**2 + b * (x2 - x1**2)**2) #+ noise_std * np.random.standard_normal()

def himmelblau(x):
    x1, x2 = x
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

objective = himmelblau

start_points = 50 #d + 1
start_x = surr.sample_in_bounds(start_points)
start_y = np.array([objective(x) for x in start_x]) 

num_iters = 30
surr.update(start_x, start_y)

for i in range(num_iters):
    x, _ = surr.optimize(acquisition='ei')
    y = himmelblau(x)
    surr.update(x, y)
    print(f"Iter: {i}: minimum: {y}")

m = 100
x1 = np.linspace(lb[0], ub[0], m)
x2 = np.linspace(lb[1], ub[1], m)
X1, X2 = np.meshgrid(x1, x2)

X = np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1)))
f = np.array([himmelblau(x) for x in X]).reshape(m, m)

fig, axs = plt.subplots(1, 2, layout='constrained', figsize=(10,5))

vmin=np.min(f)
vmax=np.max(f)
cmap = 'viridis'
locator = MaxNLocator(nbins=20, symmetric=False)
levels = locator.tick_values(vmin, vmax)
axs[0].contourf(X1, X2, f, levels=levels, cmap=cmap)

f_surr = np.array([surr(x) for x in X]).reshape(m, m)
axs[1].contourf(X1, X2, f_surr, levels=levels, cmap=cmap)
axs[1].scatter(surr.X[:, 0], surr.X[:, 1], color = 'red')

fig.savefig("rosenbrock.png")





