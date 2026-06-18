import copy
import numpy as np
import pickle

from surrogate import Surrogate
from forward_model import ForwardModel
from reverse_model import ReverseModel
from controls import ThrusterController, ControlPoint, calc_performance_metrics

from concurrent.futures import ThreadPoolExecutor

INF = float("inf")


def _validate_in_range(val, name, lo=float("-inf"), hi=float("inf")):
    if val < lo or val > hi:
        raise ValueError(f"{name} must be between {lo} and {hi}! Got: {val}")
    return val


#  TODO
# - Logging of what commands we send, and what data we get back
# - Could save all reverse + forward samples in some per-iteration folder
# - Build + update surrogate in 1-3 D
# - conditioning on normalized discharge current vector


def log_penalty(x, lb, ub, penalty_strength=5e-2):
    """Evaluate a smoothly differentiable constraint penalty function that is ~zero far from the constraints and ~inf close to them"""
    # Using logarithmic barrier function: https://en.wikipedia.org/wiki/Barrier_function
    # We should save the actual predicted metric separately from the combined objective function
    # Here's a desmos link demonstrating the functions in 1D: https://www.desmos.com/calculator/wjk4t1m2z5
    midpoint = 0.5 * (lb + ub)
    midpoint_f = -np.log(midpoint - lb) - np.log(ub - midpoint)
    eps = 1e-6
    x = np.maximum(lb * (1 + eps), np.minimum(ub * (1 - eps), x))
    penalty_per_dim = -np.log(x - lb) - np.log(ub - x) - midpoint_f
    total_penalty = penalty_strength * np.mean(penalty_per_dim)
    # print(f"{x[0]=}, {total_penalty=}")
    return total_penalty


class DiffusionController:
    def __init__(
        self,
        c0,  # Starting control values. TODO: also pass specification listing what each index is (or pass as dict)
        control_vars: list[str],  # Which controls are active (strings)
        controller: ThrusterController,  # Thruster controller
        # Forward model, mapping controls + state -> new state + data.
        forward: ForwardModel | None = None,
        surrogate: Surrogate | None = None,  # Surrogate model type, TODO: define API.
        metric=None,  # Function from data -> reals, positive definite.
        # Reverse model, maps data to several (control, state) estimates.
        reverse: ReverseModel | None = None,
        num_reverse_steps=128,
        num_reverse_samples=16,
        forwards_per_reverse=1,  # How many forward model samples to draw per reverse sample
        model_trust=1.0,  # Starting model trust parameter (default 1).
        trust_relaxation=0.5,  # Under-relaxation parameter for updating model trust.
        control_lb=None,  # Lower bounds for all control variables
        control_ub=None,  # Upper bounds for all control variables
        penalty_strength=5e-2,  # The scale factor of the logarithmic penalty function used to avoid the bounds
        cff=0.07,  # cathode flow fraction
    ):
        self.iter = 0
        self.cff = cff

        # TODO: decide on a way to decrement step_scale over time
        self.control_point = c0

        # Running list of control points and metrics
        self.cs = []
        self.zs = []

        self.control_vars = control_vars
        self.control_dim = len(control_vars)

        # Check length and contents of bounds
        self.penalty_strength = penalty_strength

        self.control_lb = (
            control_lb if control_lb else [-INF for _ in range(self.control_dim)]
        )
        self.control_ub = (
            control_ub if control_ub else [INF for _ in range(self.control_dim)]
        )
        assert len(self.control_lb) == self.control_dim, (
            f"Control lower bound must have length {self.control_dim} to match control dimension, but got {len(self.control_lb)}"
        )
        assert len(self.control_ub) == self.control_dim, (
            f"Control upper bound must have length {self.control_dim} to match control dimension, but got {len(self.control_lb)}"
        )
        assert np.all(self.control_ub >= self.control_lb), (
            f"Control upper bound must be >= lower bound for all variables. Got lb: {control_lb} and ub: {control_ub}."
        )

        self.control_lb = np.array(self.control_lb)
        self.control_ub = np.array(self.control_ub)
        self.step_scale = (self.control_ub - self.control_lb) / 4.0

        self.num_reverse_steps = num_reverse_steps
        self.num_reverse_samples = num_reverse_samples

        # Set up the three main components of the controller/optimizer
        self.controller = controller
        self.forward = forward
        self.reverse = reverse
        self.surrogate = surrogate

        # Check metric
        if metric is None:
            raise ValueError(
                "Metric must be specified! This should be a function of the data which returns a positive number."
            )

        def metric_with_constraint(y, c):
            z = metric(y)
            c_vec = self.dict_to_vec(c)
            z_prime = z + log_penalty(
                c_vec, self.control_lb, self.control_ub, self.penalty_strength
            )
            return z, z_prime

        self.metric = metric_with_constraint

        self.forwards_per_reverse = _validate_in_range(
            forwards_per_reverse, "Forwards per reverse", lo=1
        )

        # Asynchronous executor, used to spin off forward model calls
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Previous model and surrogate predicted metrics, used to update model trust
        self.z_pred_surr = None
        self.z_pred_model = None
        self.z_pred_model_future = None

        self.model_trust = _validate_in_range(model_trust, "Model trust", 0, 1)
        self.trust_relaxation = _validate_in_range(
            trust_relaxation, "Trust relaxation", 0, 1
        )

        self.metadata = {
            "control_vars": self.control_vars,
            "control_lb": list(self.control_lb),
            "control_ub": list(self.control_ub),
            "penalty_strength": self.penalty_strength,
            "trust_relaxation": self.trust_relaxation,
        }

        self.iter_logs = []
        self.current_iter_log = None

        self.surrogate_tol = 0.01

    def load_from_file(self, filename):
        # TODO: reconstruct controller state from output file
        pass

    def save_to_file(self, filename):
        out_dict = {
            "metadata": self.metadata,
            "iterations": self.iter_logs,
        }

        with open(filename, "wb") as fd:
            pickle.dump(out_dict, fd)

    def dict_to_vec(self, control_dict: dict):
        """
        Convert a control setpoint from a dictionary to a vector.
        This vector only contains the active control point.
        """
        v = np.array([control_dict[k] for k in self.control_vars])
        return v

    def vec_to_dict(self, control_vec: list | np.ndarray):
        """
        Convert a control setpoint from a vector representation to a dictionary.
        Only the active control variable is represented in the vector, so the inactive
        variables are filled in from the current control setpoint.
        """
        d = copy.deepcopy(self.control_point)
        for k, v in zip(self.control_vars, control_vec):
            d[k] = v
        return d

    def dict_to_struct(self, control_dict: dict) -> ControlPoint:
        return ControlPoint(
            magnetic_field_scale=control_dict["magnetic_field_scale"],
            discharge_voltage_v=control_dict["discharge_voltage_v"],
            anode_mass_flow_rate_kg_s=control_dict["anode_mass_flow_rate_kg_s"],
            cathode_flow_fraction=self.cff,
        )

    def command_thruster(self, client, **data_args):
        """Control thruster to the given control point and take data"""
        # self.control_point is always a dictionary containing the full specification of the current control setpoint
        # (including but not limited to the active optimization variables)
        control_pt_struct = self.dict_to_struct(self.control_point)

        self.controller.control_to(client, control_pt_struct)
        data = self.controller.take_data(client, **data_args)

        # Evaluate metric on data.
        # The metric returns two values -- a raw metric and one that incorporates constraints.
        # Here, we're interested in the former.
        z, _ = self.metric(data, self.control_point)
        perf = calc_performance_metrics(data, control_pt_struct)

        if self.current_iter_log is not None:
            # Log this stage
            self.current_iter_log["stages"]["command_thruster"] = {
                "control_point": self.control_point,
                "data": data,
                "perf": perf,
                "metric": z,
            }

        return data, perf, z

    def update_model_trust(self, z):
        """
        Update the model trust parameter based on the previous iteration's model and surrogate predictions.
        """
        # Await results of forward model from before and average the metrics
        if self.z_pred_model_future is not None:
            z_pred_model_results = self.z_pred_model_future.result()
            self.z_pred_model = 0.0
            count = 0
            for result in z_pred_model_results:
                if result is None:
                    continue
                _, yk = result
                self.z_pred_model += self.metric(yk, self.control_point)[0]
                count += 1
            self.z_pred_model /= count

        if self.z_pred_model is None and self.z_pred_surr is None:
            # If we're in the first loop, we don't have previous predictions,
            # so we can't update the trust parameter
            pass
        else:
            if self.z_pred_surr is None:
                # No surrogate specified, we have to trust the model
                self.model_trust = 1.0
            elif self.z_pred_model is None:
                # No model, we have to trust the surrogate
                self.model_trust = 0.0
            else:
                # Use inverse distance weighting to interpolate between surrogate and modeling
                # The distance is evaluated as the difference between predicted and observed
                # z for a specified control action
                beta = self.trust_relaxation
                dz_surr = np.abs(z - self.z_pred_surr)
                dz_model = np.abs(z - self.z_pred_model)
                new_trust = (1.0 / dz_model) ** 2 / (
                    1.0 / dz_model**2 + 1.0 / dz_surr**2
                )
                self.model_trust = beta * new_trust + (1 - beta) * self.model_trust

        if self.current_iter_log is not None:
            self.current_iter_log["stages"]["update_model_trust"] = {
                "z_pred_surr": self.z_pred_surr,
                "z_pred_model": self.z_pred_model,
                "T": self.model_trust,
            }

        return self.model_trust

    def get_surrogate_proposed_control(self, c: np.ndarray, z):
        rand_pt = np.array(
            [
                np.random.uniform(lb, ub)
                for (lb, ub) in zip(self.control_lb, self.control_ub)
            ]
        )

        if self.surrogate is not None:
            # Update surrogate model with new data point
            self.surrogate.update([c], [z])
            # Perform local optimization on surrogate
            # to find optimal control location
            if self.surrogate.is_trained:
                c_surr, _ = self.surrogate.optimize(tol=self.surrogate_tol)
            else:
                c_surr = rand_pt
        else:
            c_surr = rand_pt

        if self.current_iter_log is not None:
            self.current_iter_log["stages"]["update_surrogate"] = {
                "c_surr": c_surr.tolist()
            }

        return c_surr

    def get_model_proposed_control(self, c_surr, y):
        control_vec = self.dict_to_vec(self.control_point)
        if self.reverse is not None and self.forward is not None:
            # Merge data dictionary and control point dictionary to condition the diffusion model
            condition_input = self.control_point | y
            # TODO: Condition diffusion model on discharge current
            # TODO: 1. normalize fourier signal
            # TODO: 2. pass as conditioning vector
            # TODO: should we use the current point or the best point to initialize this?
            reverse_samples = self.reverse(
                condition_input,
                num_samples=self.num_reverse_samples,
                num_steps=self.num_reverse_steps,
            )

            # Propose one or more control actions
            proposed_controls = []
            for xk in reverse_samples:
                for _ in range(self.forwards_per_reverse):
                    # Proposed control action is a mixture of surrogate direction and random noise
                    # Balances exploration / exploitation
                    rand_direction = np.random.standard_normal(self.control_dim)
                    surr_direction = c_surr - control_vec
                    c_proposed = control_vec + self.step_scale * (
                        (1 - self.model_trust) * surr_direction + rand_direction
                    )
                    proposed_controls.append([xk, self.vec_to_dict(c_proposed)])

            # Evaluate forward model for each state estimate / control action pair
            # We do this asynchronously but immediately await the result
            # The async part is thus not really necessary, but I am leaving it in in case we
            # might want to interleave additional work in the future.
            future = self.executor.submit(self.forward, proposed_controls)
            forward_samples = future.result()

            # Eval metrics and weight control proposals based on metrics
            # This could be in the previous loop if serial, but
            # that loop should be made parallel so I'm keeping it separate
            numerator = np.zeros(self.control_dim)
            denominator = 0.0
            sim_metrics = []
            for forward_sample, (xk, ck) in zip(forward_samples, proposed_controls):
                if forward_sample is None:
                    # This occurs for simulation failures and other invalid states
                    continue

                (_, fourier) = forward_sample

                # TODO: generalize this for other objectives
                yk = {"mean_current_A": fourier[0], "rms_current_A": fourier[1]}

                # Use version of metric with constraint.
                ck_vec = self.dict_to_vec(ck)
                z_base, z_with_constraints = self.metric(yk, ck)
                sim_metrics.append(z_base)
                numerator += ck_vec / z_with_constraints**2
                denominator += 1.0 / z_with_constraints**2

            # Get final model-proposed control point
            c_model = numerator / denominator
        else:
            c_model = np.zeros(self.control_dim)
            proposed_controls = []
            sim_metrics = []
            reverse_samples = []

        if self.current_iter_log is not None:
            self.current_iter_log["stages"]["get_model_proposed_control"] = {
                "c_proposed": [
                    list(self.dict_to_vec(c)) for (_, c) in proposed_controls
                ],
                "z_proposed": sim_metrics,
                "c_model": list(c_model),
            }

        return c_model, reverse_samples

    def get_final_control(self, c_surr, c_model):
        # Once we have the model and surrogate-proposed controls in hand,
        # we can determine the final control action by interpolating between
        # the two based on model trust
        if self.surrogate is None:
            c_final = c_model
        elif self.forward is None or self.reverse is None:
            c_final = c_surr
        else:
            c_final = (1 - self.model_trust) * c_surr + self.model_trust * c_model

        # Ensure c_final does not violate constraints
        c_final = np.maximum(self.control_lb, np.minimum(self.control_ub, c_final))

        if self.current_iter_log is not None:
            self.current_iter_log["stages"]["get_final_control"] = {
                "c_final": list(c_final)
            }

        return c_final

    def step(self, client):
        # Start a new log for this iteration
        # TODO: keep track of best sample point
        # TODO: restart based on what stage we were at in the output file
        assert self.forward is not None
        self.iter_logs.append({})
        self.current_iter_log = self.iter_logs[-1]

        self.current_iter_log["iter"] = self.iter
        self.current_iter_log["step_scale"] = list(self.step_scale)
        self.current_iter_log["model_trust"] = self.model_trust
        self.current_iter_log["control_point"] = list(self.control_point)
        self.current_iter_log["stages"] = {}

        # command thruster to current control point
        _, y, z = self.command_thruster(client)
        control_vec = self.dict_to_vec(self.control_point)

        # Save the current control point and experimental metric
        self.cs.append(control_vec)
        self.zs.append(z)

        # Update the model trust
        self.update_model_trust(z)

        # Propose control actions using surrogate and model, then combine them to get the
        # final proposed control point
        c_surr = self.get_surrogate_proposed_control(control_vec, z)

        c_model, reverse_samples = self.get_model_proposed_control(c_surr, y)
        c_final = self.get_final_control(c_surr, c_model)

        # Predict surrogate output at this point
        self.z_pred_surr = self.surrogate(c_final) if self.surrogate else None

        # Predict mean model output asynchronously (so we can simultaneously command the thruster)
        final_controls = [(xk, c_final) for xk in reverse_samples]
        self.z_pred_model_future = self.executor.submit(self.forward, final_controls)

        self.iter += 1

        # Update final proposed control action and return it
        self.control_point = self.vec_to_dict(c_final)
        return self.control_point, z
