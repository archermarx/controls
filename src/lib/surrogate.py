import numpy as np
from scipy.optimize import minimize, Bounds
from smt.sampling_methods import LHS
from smt.surrogate_models import KRG, KPLS
from scipy.stats import norm
from scipy.spatial.distance import cdist
from typing import Literal, get_args

AcquisitionFunction = Literal["ei", "eig", "mean"]
CovarianceKernel = Literal["matern32", "matern52", "squar_exp"]
ModelType = Literal["KRG", "KPLS"]

class Surrogate:
    def __init__(
        self,
        bounds,
        model_type: ModelType = "KRG",
        kernel: CovarianceKernel = "matern32",
        optimize_restarts: int = 10,
        hyperopt_restarts: int = 10,
        acquisition: AcquisitionFunction ="ei",
        noise_floor: float = 1e-3,
    ):
        self.lb, self.ub = np.array(bounds[0]), np.array(bounds[1])
        if len(self.ub) != len(self.lb):
            raise ValueError("Upper and lower bounds must have the same length!")

        self.dim = len(self.lb)
        self.min_points = max(2, self.dim + 1)
        self.kernel = kernel
        self.optimize_restarts = optimize_restarts

        self.model_type = model_type
        self.hyperopt_restarts = hyperopt_restarts
        if self.model_type == "KPLS" and self.kernel != "squar_exp":
            raise ValueError("Only square exponential kernel ('squar_exp') supported with KPLS")

        self.allowed_acquisitions = {"mean", "ei", "eig", "ei_eig"}
        if acquisition not in self.allowed_acquisitions:
            raise ValueError(f"acquisition must be in {self.allowed_acquisitions}")

        self.acquisition = acquisition
        self.noise_floor = noise_floor

        self.X = np.zeros((0, 2))
        self.Y = np.zeros(0)

        self.model = None
        self.is_trained = False

        self.best_x = np.zeros(self.dim)
        self.best_y = np.inf

    def to_dict(self):
        return {
            "X": self.X,
            "Y": self.Y,
            "acquisition": self.acquisition,
            "optimize_restarts": self.optimize_restarts,
            "kernel": self.kernel,
            "lb": self.lb,
            "ub": self.ub,
            "noise_floor": self.noise_floor,
            "optimal_theta": self.model.optimal_theta if self.model else None,
            "optimal_noise": self.model.optimal_noise if self.model else None
        }

    @staticmethod
    def from_dict(d):
        surr = Surrogate(
            bounds = (d["lb"], d["ub"]),
            kernel = d["kernel"],
            optimize_restarts=d["optimize_restarts"],
            acquisition = d["acquisition"],
            noise_floor = d["noise_floor"],
        )
        surr.update(d["X"], d["Y"])
        return surr

    def __call__(self, x) -> float:
        assert self.model is not None
        x = self._as_vector(x)

        if len(self.Y) == 0:
            return 0.0

        if not self.is_trained:
            return float(min(self.Y))

        x_scaled = self._scale(x.reshape(1, -1))
        z = self.model.predict_values(x_scaled)
        return float(z[0, 0])

    def sample_in_bounds(self, n):
        bounds = [(lb, ub) for lb, ub in zip(self.lb, self.ub)]
        return LHS(xlimits=np.array(bounds))(n)

    def update(self, x, y) -> None:
        X = np.atleast_2d(x)
        Y = np.atleast_1d(y)

        # Check for shapes
        assert len(X.shape) == 2
        assert len(Y.shape) == 1
        assert X.shape[1] == self.dim

        if len(self.X) == 0:
            self.X = X
            self.Y = Y
        else:
            self.X = np.concat((self.X, X))
            self.Y = np.concat((self.Y, Y))
        self._fit()

    def _fit(self):
        # Keep track of best point
        self.Y = np.array(self.Y)
        self.X = np.array(self.X)
        self.best_ind = np.argsort(self.Y)
        self.best_y = self.Y[self.best_ind]
        self.best_x = self.X[self.best_ind]

        X_scaled = self._scale(self.X)
        X_unique, Y_unique = self._remove_duplicate_points(X_scaled, self.Y)

        if len(Y_unique) < self.min_points:
            self.is_trained = False
            return

        # Multistart hyperparameter optimization
        theta_bounds = [5e-2, 1.0]
        if self.model_type == "KRG":
            model = KRG(
                theta0=[0.5 * (theta_bounds[1] - theta_bounds[0])],
                corr=self.kernel, 
                theta_bounds=theta_bounds,
                print_global=False,
                eval_noise=True,
                hyper_opt="Cobyla",
                n_start=self.hyperopt_restarts,
            )
        elif self.model_type == "KPLS":
            model = KPLS(
                theta0=[0.5 * (theta_bounds[1] - theta_bounds[0])],
                theta_bounds=theta_bounds,
                n_comp=self.dim,
                print_global=False,
                eval_noise=True,
                hyper_opt="Cobyla",
                corr=self.kernel,
                n_start=self.hyperopt_restarts,
            )
        else:
            assert False

        model.set_training_values(X_unique, Y_unique.reshape(-1, 1))
        model.train()

        self.model = model
        self.is_trained = True

    def optimize(self, acquisition=None, method: str = "L-BFGS-B", tol=1e-2) -> tuple[np.ndarray, float]:
        assert self.model is not None
        mode = self.acquisition if acquisition is None else acquisition

        if mode not in get_args(AcquisitionFunction):
            raise ValueError(f"acquisition must be in {get_args(AcquisitionFunction)}")

        if not self.is_trained:
            x0 = 0.5 * (self.ub - self.ub)
            return x0, self(x0)

        # Generate random start locations using LHS sampling
        start_locs = self.sample_in_bounds(self.optimize_restarts)

        # Perform optimization
        objective = self._acquisition_objective(mode)
        optim_best_x = start_locs[0]
        optim_best_y = np.inf
        for start in start_locs:
            result = minimize(
                objective,
                start,
                method=method,
                bounds=Bounds(self.lb, self.ub) # type:ignore
            )

            opt_x = result.x
            opt_y = result.fun
            if opt_y < optim_best_y:
                optim_best_y = opt_y
                optim_best_x = opt_x

        # pick random point if too close to existing point
        X_scaled = self._scale(self.X)
        best_x_scaled = self._scale(optim_best_x)

        dists = cdist(X_scaled, [best_x_scaled])
        argmin_dist = np.argmin(dists)
        if dists[argmin_dist] < tol:
            #print(f"{optim_best_x=} (closest={self.X[argmin_dist]})")
            if acquisition == "ei":
                print(f"Optimizing using EIG")
                return self.optimize(acquisition="eig", tol=tol)
            else:
                print(f"Using random point")
                optim_best_x = np.array([
                    np.random.uniform(self.lb[i], self.ub[i])
                    for i in range(self.dim)
                ])

        return optim_best_x, optim_best_y

    def variance(self, x) -> float:
        assert self.model is not None
        x = self._as_vector(x)

        if not self.is_trained:
            return float("inf")

        x_scaled = self._scale(x.reshape(1, -1))
        var = self.model.predict_variances(x_scaled)
        return max(0.0, float(var[0, 0]))

    def mean_and_std(self, x) -> tuple[float, float]:
        """
        Return the Kriging predicted mean and standard deviation at x.
        """
        assert self.model is not None

        x = self._as_vector(x)

        if not self.is_trained:
            raise RuntimeError(
                "The surrogate must be trained before predicting uncertainty."
            )

        x_scaled = self._scale(x.reshape(1, -1))

        mean = float(self.model.predict_values(x_scaled)[0, 0])
        variance = float(self.model.predict_variances(x_scaled)[0, 0])

        variance = max(variance, self.noise_floor**2)
        std = np.sqrt(variance)

        return mean, float(std)

    def expected_improvement(self, x, y_best=None) -> float:
        """
        Expected Improvement acquisition value for minimization.
        A larger value means x is a more useful control point to evaluate next.
        """
        if not self.is_trained:
            return 0.0

        assert self.model is not None

        if y_best is None:
            # Calculate incumbent value: use surrogate prediction of all sample points
            # TODO: optimize this so we don't calculate min every time
            y_best = self.model.predict_values(np.array(self.X)).min()

        # --- Basic EI: Only account for variance in surrogate ---
        mean, std = self.mean_and_std(x)
        improvement = y_best - mean
        gamma = improvement / std
        ei = improvement * norm.cdf(gamma) + std * norm.pdf(gamma)

        # --- Augmented EI (Huang et al. 2006): Discount EI when estimated measurment noise is large ---
        noise_var = self.model.optimal_noise
        discount = 1 - np.sqrt(noise_var) / np.sqrt(std**2 + noise_var)
        ei_aug = ei * discount

        return max(0.0, ei_aug)

    def expected_information_gain(self, x) -> float:
        """
        Expected Information Gain
        A larger value means x is likely to contain information that improves the surrogate quality.
        """
        assert self.model is not None
        _, std = self.mean_and_std(x)
        pred_var = std**2
        noise_var = np.atleast_1d(self.model.optimal_noise + self.noise_floor)[0]

        eig = 0.5 * np.log(1 + pred_var / noise_var)
        return float(eig)

    def _acquisition_objective(self, acquisition):
        if acquisition == "mean":
            return self
        if acquisition == "ei":
            # Calculate incumbent value: use surrogate prediction of all sample points
            assert self.model is not None
            y_best = self.model.predict_values(np.array(self.X)).min()
            return lambda x: -self.expected_improvement(x, y_best)
        if acquisition == "eig":
            return lambda x: -self.expected_information_gain(x)

        raise ValueError("acquisition must be 'mean', 'ei', or 'eig'")

    def _remove_duplicate_points(self, X, Y):
        rounded = np.round(X, decimals=12)
        X_unique, inverse = np.unique(rounded, axis=0, return_inverse=True)

        Y_unique = np.zeros(len(X_unique))
        counts = np.zeros(len(X_unique))

        for i, group in enumerate(inverse):
            Y_unique[group] += Y[i]
            counts[group] += 1

        Y_unique /= counts
        return X_unique, Y_unique

    def _scale(self, X):
        # Scale inputs to range (0, 1)
        width = self.ub - self.lb
        width[width == 0.0] = 1.0
        return (X - self.lb) / width

    def _as_vector(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.dim:
            raise ValueError(f"Expected dimension {self.dim}, got {x.size}")
        return x
    
    def plot_1d_on_axis(
        self,
        ax,
        ground_truth=None,
        num_points=400,
        confidence=2.0,
        xlabel="Control c",
        ylabel="Metric z",
        title="Kriging surrogate",
        ground_truth_label="Ground truth",
        extension=0,
    ):
        """
        Plot a one-dimensional Kriging surrogate and its uncertainty.

        Parameters
        ----------
        ground_truth:
            Either:

            1. A callable:
                ground_truth(c) -> z

            2. A tuple containing existing data arrays:
                (ground_truth_controls, ground_truth_metrics)

            3. None, in which case no ground-truth curve is shown.

        num_points:
            Number of points used to draw the surrogate curve.

        errorbar_points:
            Number of locations where uncertainty error bars are drawn.

        confidence:
            Number of predictive standard deviations used for the uncertainty.
            confidence=2 approximately corresponds to mean ± 2 standard deviations.

        filename:
            Optional path where the figure is saved.

        show:
            Whether to display the plot interactively.
        """

        if self.dim != 1:
            raise ValueError(
                "plot_1d() only works for a one-dimensional surrogate. "
                "For a multidimensional surrogate, plot a one-dimensional slice."
            )

        if not self.is_trained:
            raise RuntimeError(
                "The surrogate must be trained before it can be plotted."
            )

        # Real control points and measured metric values used to train KRG
        observed_c = self.X[:, 0]
        observed_z = np.asarray(self.Y, dtype=float)

        # Decide what control range should be plotted
        lower, upper = self.lb[0], self.ub[0]

        # Dense grid used to draw the surrogate function
        c_grid = np.linspace(lower - extension, upper + extension, num_points)
        c_grid_matrix = c_grid.reshape(-1, 1)
        c_grid_scaled = self._scale(c_grid_matrix)

        assert self.model is not None

        # Kriging predicted mean and variance
        predicted_mean = self.model.predict_values(c_grid_scaled).reshape(-1)
        predicted_variance = self.model.predict_variances(c_grid_scaled).reshape(-1)

        # Small negative variances can occur from numerical roundoff
        predicted_variance = np.maximum(predicted_variance, 0.0)

        # Error bars must use standard deviation, not raw variance
        predicted_std = np.sqrt(predicted_variance)
        uncertainty = confidence * predicted_std

        confidence_lower = predicted_mean - uncertainty
        confidence_upper = predicted_mean + uncertainty

        # Plot the Kriging mean function
        ax.plot(
            c_grid,
            predicted_mean,
            label="Kriging predicted mean",
            linewidth=2,
        )

        if self.acquisition == "ei":
            c_ei, z_ei = self.optimize(acquisition="ei")

            ax.scatter(
                c_ei[0],
                z_ei,
                marker="D",
                s=90,
                label="EI-selected next point",
                zorder=6,
            )

        ax.fill_between(
            c_grid,
            confidence_lower,
            confidence_upper,
            alpha=0.15,
            label=(
                f"Mean ± {confidence:g} standard deviations"
            ),
        )

        # Plot the actual control/metric observations used for training
        ax.scatter(
            observed_c,
            observed_z,
            marker="o",
            s=55,
            label="Observed control points",
            zorder=4,
        )

        # Plot the ground-truth function if one was provided
        if ground_truth is not None:
            if callable(ground_truth):
                ground_truth_c = c_grid
                ground_truth_z = np.asarray(
                    [ground_truth(c) for c in ground_truth_c],
                    dtype=float,
                )
            else:
                if len(ground_truth) != 2:
                    raise ValueError(
                        "ground_truth must be either a callable or "
                        "a tuple of (control_values, metric_values)."
                    )

                ground_truth_c = np.asarray(
                    ground_truth[0],
                    dtype=float,
                ).reshape(-1)

                ground_truth_z = np.asarray(
                    ground_truth[1],
                    dtype=float,
                ).reshape(-1)

                if ground_truth_c.size != ground_truth_z.size:
                    raise ValueError(
                        "Ground-truth control and metric arrays "
                        "must have the same length."
                    )

            ax.plot(
                ground_truth_c,
                ground_truth_z,
                linestyle="--",
                linewidth=2,
                label=ground_truth_label,
            )

        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid(True, alpha=0.3)
        ax.legend()