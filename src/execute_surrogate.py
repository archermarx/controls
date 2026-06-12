import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

import lib.controls as controls
import lib.labview as labview
from lib.surrogate import Surrogate

parser = argparse.ArgumentParser()

parser.add_argument("--cal-file", "-c", type=Path, help="The path to the thruster calibration file")
parser.add_argument("--setpoint", "-s", type=Path, required=True)
parser.add_argument("--control-vars", type=str, required=True)
parser.add_argument("--bounds", type=str, required=True)
parser.add_argument("--data", "-d", type=lambda s: [item.strip() for item in s.split(",") if item.strip()], default=["dmm", "oscope"],)
parser.add_argument("--num-steps", "-n", type=int, default=25)
parser.add_argument("--dwell-time", "-t", type=int, default=5)
parser.add_argument("--gas", "-g", type=str, choices=["Xe", "Kr", "Ar"], default="Kr")

parser.add_argument("--host-ip", type=str, default=labview.LABVIEW_IP)
parser.add_argument("--port", type=int, default=labview.LABVIEW_PORT)
parser.add_argument("--verbose", "-v", action="store_true")

parser.add_argument("--optimize-restarts", type=int, default=25)
parser.add_argument("--acquisition", type=str, choices=["ei", "mean", "ei_eig"], default="ei")
parser.add_argument("--plot-acquisition", type=str, choices=["ei", "mean", "ei_eig"], default=None)
parser.add_argument("--seed", type=int, default=1)

parser.add_argument("--max-current-offset-A", type=float, default=2.0)
parser.add_argument("--max-current-offset-frac", type=float, default=0.25)
parser.add_argument("--allow-current-mismatch", action="store_true")

parser.add_argument("--output", "-o", type=Path, default=Path("."))
parser.add_argument("--prefix", "-p", type=str, default="surrogate")
parser.add_argument("--reset-at-end", action="store_true")
parser.add_argument("--interactive", action="store_true")

parser.add_argument("--remote-dir", type=Path)

def compute_rms_amplitude_master(data):
    dmm: labview.KeysightDMMReadings = data["dmm"]
    anode_current: labview.OscopeReadings = data["oscope"]["Anode Current"]

    time, current = anode_current.waveform.time_values(), anode_current.waveform.y_values()
    mean_oscope = np.mean(current)
    mean_dmm = dmm.current
    current_centered = current - mean_oscope

    # centered rms = sqrt(mean((I - I_mean)^2))
    rms_current = np.sqrt(np.mean((current_centered)**2))
    return rms_current, rms_current/mean_dmm

def rms_amplitude_raw(data, *args, **kwargs):
    return compute_rms_amplitude_master(data)[0]

def rms_amplitude_pct(data, *args, **kwargs):
    return compute_rms_amplitude_master(data)[1]

def parse_control_vars(text):
    return [x.strip() for x in text.split(",") if x.strip()]

def parse_bounds(text):
    bounds = []

    for item in text.split(","):
        item = item.strip()

        if ":" not in item:
            raise ValueError(f"Bad bound '{item}'. Use lower:upper format.")

        lo, hi = item.split(":", maxsplit=1)
        lo = float(lo)
        hi = float(hi)

        if lo >= hi:
            raise ValueError(f"Bad bound '{item}': lower must be less than upper.")

        bounds.append((lo, hi))

    return bounds


def clip_to_bounds(c, bounds):
    return np.array(
        [
            np.clip(value, lo, hi)
            for value, (lo, hi) in zip(c, bounds)
        ],
        dtype=float,
    )

def read_setpoint(path):
    with open(path, "rb") as fd:
        return controls.ControlPoint.model_validate_json(fd.read())

def setpoint_to_vector(setpoint, control_vars):
    return np.array(
        [float(getattr(setpoint, name)) for name in control_vars],
        dtype=float,
    )

def vector_to_setpoint(base_setpoint, control_vars, c):
    setpoint = base_setpoint.model_copy(deep=True)
    for name, value in zip(control_vars, c):
        setattr(setpoint, name, float(value))
    return setpoint

def plot_acquisition(ax, surrogate, bounds, mode):
    lb, ub = bounds[0]
    x = np.linspace(lb, ub, 100)

    if mode == "mean":
        y = [surrogate([_x]) for _x in x]
        ax.plot(x, y, color="red")
        ax.set(title="Predicted mean", xticklabels=[], xlim=(lb, ub))
        return

    if mode == "ei":
        y = [surrogate.expected_improvement([_x]) for _x in x]
        ax.plot(x, y, color="red")
        ax.set(title="Expected improvement", xticklabels=[], xlim=(lb, ub))
        return

    if mode == "ei_eig":
        y = []

        for _x in x:
            if surrogate.is_near_bounds([_x]):
                y.append(surrogate.expected_information_gain([_x]))
            else:
                y.append(surrogate.expected_improvement([_x]))

        ax.plot(x, y, color="red")

        edge = surrogate.boundary_eig_frac * (ub - lb)
        ax.axvspan(lb, lb + edge, alpha=0.15)
        ax.axvspan(ub - edge, ub, alpha=0.15)

        ax.set(title="Expected improvement / information gain", xticklabels=[], xlim=(lb, ub))
        return

    raise ValueError(f"Unknown plot acquisition mode: {mode}")

def main(args):
    control_vars = parse_control_vars(args.control_vars)
    bounds = parse_bounds(args.bounds)

    if len(control_vars) != len(bounds):
        raise ValueError("Number of control variables must match number of bounds.")

    dim = len(control_vars)


    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)

    base_setpoint = read_setpoint(args.setpoint)

    np.random.seed(1234)
    initial_controls = [setpoint_to_vector(base_setpoint, control_vars)]
    initial_controls += [np.array([np.random.uniform(lb, ub) for (lb, ub) in bounds]) for _ in range(dim)]
    print(f"{initial_controls=}")

    metric_fn = rms_amplitude_pct

    surrogate = Surrogate(
        dim=dim,
        bounds=bounds,
        min_points=dim+1,
        optimize_restarts=args.optimize_restarts,
        acquisition=args.acquisition,
        seed=args.seed,
    )

    controller = controls.ThrusterController(
        args.cal_file, propellant=args.gas, verbose=args.verbose
    )

    c_initial_raw = setpoint_to_vector(base_setpoint, control_vars)
    c_initial = clip_to_bounds(c_initial_raw, bounds)

    if not np.allclose(c_initial_raw, c_initial):
        print("Warning: initial setpoint was outside the supplied bounds.")
        print(f"Raw initial control:     {c_initial_raw}")
        print(f"Clipped initial control: {c_initial}")


    print("Starting surrogate control.")
    print(f"Control variables: {control_vars}")
    print(f"Bounds: {bounds}")
    print(f"Data sources: {args.data}")
    print(f"Steps: {args.num_steps}")
    print()

    print("Initial control vectors:")
    for i, c in enumerate(initial_controls):
        source = "original setpoint" if i == 0 else "hard-coded perturbation"
        print(f"    {i + 1}: {source}: {c}")
    print()

    log = []
    should_exit=False

    with labview.LabViewClient(host=args.host_ip, port=args.port) as client:
        for step in range(args.num_steps):
            if step < dim+1:
                c_current = initial_controls[step]
                point_source = "initial_setpoint" if step == 0 else "initial_perturbation"
                z_pred = np.nan

            else:
                if not surrogate.is_trained:
                    print()
                    print("Initial points are finished, but surrogate is not trained.")
                    print("Stopping safely instead of commanding more points.")
                    print("This usually means some initial points became duplicates after clipping.")
                    break

                c_current, z_pred = surrogate.optimize(
                    acquisition=args.acquisition,
                )

                c_current = clip_to_bounds(c_current, bounds)
                point_source = "surrogate"

            setpoint = vector_to_setpoint(
                base_setpoint,
                control_vars,
                c_current,
            )

            print()
            print(f"Step {step + 1}/{args.num_steps}")
            print(f"Point source: {point_source}")
            print(f"Commanding: {setpoint}")
            if args.interactive:
                while True:
                    inp = input(f"Continue? (y/n): ")
                    if inp.casefold() == "y":
                        break
                    elif inp.casefold() == "n":
                        should_exit=True
                        break

            if should_exit:
                print("Exiting!")
                break

            controller.control_to(setpoint, client)

            data = controller.take_data(
                client,
                delay=args.dwell_time,
                sources=args.data,
            )

            z_actual = metric_fn(data)

            mean_current = data["dmm"].current
            rms_pct = rms_amplitude_pct(data)
            rms_raw = rms_amplitude_raw(data)
            print(f"Mean: {mean_current:.3f} A, RMS Amplitude: {rms_raw:.3f} A ({rms_pct*100:.2f}%)")

            surrogate.update(c_current, z_actual)
            if surrogate.is_trained:
                if dim == 1:
                    fig, axs = plt.subplots(2,1, layout='constrained', figsize=(6,6))
                    surrogate.plot_1d_on_axis(axs[1])

                    plot_mode = args.acquisition if args.plot_acquisition is None else args.plot_acquisition
                    plot_acquisition(
                        axs[0],
                        surrogate,
                        bounds,
                        plot_mode,
                    )

                    fig.savefig("surrogate.png")
                    plt.close(fig)

            sample = {
                "step": step + 1,
                "status": "ok",
                "point_source": point_source,
                "z_actual": z_actual,
                "z_pred": z_pred,
                "control_vars": control_vars,
                "control_vector": c_current,
                "setpoint": setpoint.model_dump(),
                "data": data,
                "surrogate_trained": surrogate.is_trained,
                "num_surrogate_points": len(surrogate.Y),
            }

            log.append(sample)

            out_file = output_dir / f"{args.prefix}_{step + 1:03d}.pkl"
            log_file = output_dir / f"{args.prefix}_log.pkl"

            with open(out_file, "wb") as fd:
                pickle.dump(sample, fd)

            with open(log_file, "wb") as fd:
                pickle.dump(log, fd)

            print(f"z = {z_actual:.6g}")
            print(f"Surrogate trained: {surrogate.is_trained}")
            print(f"Number of surrogate points: {len(surrogate.Y)}")
            print(f"Saved: {out_file}")

        if args.reset_at_end:
            print()
            print("Resetting to base setpoint.")
            controller.control_to(base_setpoint, client)

    print()
    print("Done.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)