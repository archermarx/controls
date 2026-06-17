import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import lib.controls as controls
import lib.labview as labview
from lib.surrogate import Surrogate

from analyze_surrogate import plot_surrogate

parser = argparse.ArgumentParser()

parser.add_argument("--cal-file", "-c", type=Path, help="The path to the thruster calibration file")
parser.add_argument("--setpoint", "-s", type=Path, required=True)
parser.add_argument("--control-vars", type=str, required=True)
parser.add_argument("--bounds", type=str, required=True)
parser.add_argument("--data", "-d", type=lambda s: [item.strip() for item in s.split(",") if item.strip()], default=None)
parser.add_argument("--num-steps", "-n", type=int, default=25)
parser.add_argument("--dwell-time", "-t", type=int, default=5)
parser.add_argument("--gas", "-g", type=str, choices=["Xe", "Kr", "Ar"], default="Kr")

parser.add_argument("--host-ip", type=str, default=labview.LABVIEW_IP)
parser.add_argument("--port", type=int, default=labview.LABVIEW_PORT)
parser.add_argument("--verbose", "-v", action="store_true")

parser.add_argument("--optimize-restarts", type=int, default=25)
parser.add_argument("--acquisition", type=str, choices=["ei", "eig", "mean"], default="ei")

parser.add_argument("--max-current-offset-A", type=float, default=2.0)
parser.add_argument("--max-current-offset-frac", type=float, default=0.25)
parser.add_argument("--allow-current-mismatch", action="store_true")

parser.add_argument("--output", "-o", type=Path, default=Path("."))
parser.add_argument("--prefix", "-p", type=str, default="surrogate")
parser.add_argument("--reset-at-end", action="store_true")
parser.add_argument("--interactive", action="store_true")
parser.add_argument("--objective", type=str, default="rms")
parser.add_argument("--restart", type=str)

parser.add_argument("--remote-dir", type=Path)

def compute_rms_amplitude_master(data, setpoint):
    dmm = data["dmm"]
    anode_current = labview.OscopeReadings.from_dict(data["oscope"]["Anode Current"])
    _, current = anode_current.waveform.time_values(), anode_current.waveform.y_values()
    mean_oscope = np.mean(current)
    mean_dmm = dmm["current"]
    current_centered = current - mean_oscope

    # centered rms = sqrt(mean((I - I_mean)^2))
    rms_current = np.sqrt(np.mean((current_centered)**2))
    return rms_current, rms_current/mean_dmm

def rms_amplitude_raw(data, setpoint, *args, **kwargs):
    return compute_rms_amplitude_master(data, setpoint)[0]

def rms_amplitude_pct(data, setpoint, *args, **kwargs):
    return compute_rms_amplitude_master(data, setpoint)[1]

def compute_efficiency(data, setpoint: controls.ControlPoint):
    dmm: dict = data["dmm"]
    thrust_mN: float = data["thrust"]["thrust_mN"]
    thrust_N = thrust_mN / 1000
    current_A = dmm["current"]
    mdot = setpoint.anode_mass_flow_rate_kg_s
    vd = setpoint.discharge_voltage_v
    return 0.5 * thrust_N**2 / mdot / vd / current_A

def thrust_to_power(data, setpoint: controls.ControlPoint):
    dmm: dict = data["dmm"]
    vd = setpoint.discharge_voltage_v
    dmm: dict = data["dmm"]
    thrust_mN: float = data["thrust"]["thrust_mN"]
    thrust_N = thrust_mN / 1000
    current_A = dmm["current"]
    vd = setpoint.discharge_voltage_v
    power_kW = vd * current_A
    return thrust_N / power_kW

def efficiency_obj(data, setpoint):
    return (1 - compute_efficiency(data, setpoint))
    
def thrust_to_power_obj(data, setpoint: controls.ControlPoint):
    return -thrust_to_power(data, setpoint)

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

def main(args):
    control_vars = parse_control_vars(args.control_vars)
    bounds = parse_bounds(args.bounds)

    if len(control_vars) != len(bounds):
        raise ValueError("Number of control variables must match number of bounds.")

    dim = len(control_vars)

    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)
    base_setpoint = read_setpoint(args.setpoint)

    if args.objective == "rms":
        metric_fn = rms_amplitude_pct
    elif args.objective == "efficiency":
        metric_fn = efficiency_obj
    elif args.objective == "thrust_to_power":
        metric_fn = thrust_to_power_obj

    lb, ub = [b[0] for b in bounds], [b[1] for b in bounds]

    if args.restart is not None:
        with open(args.restart, "rb") as fd:
            restart = pickle.load(fd)
        initial_controls = []
        surrogate = Surrogate.from_dict(restart["surrogate"])
        start_step = restart["step"]
    else:
        start_step = 0
        np.random.seed(4321)
        initial_controls = [setpoint_to_vector(base_setpoint, control_vars)]
        initial_controls += [np.array([np.random.uniform(lb, ub) for (lb, ub) in bounds]) for _ in range(dim)]
        print(f"{initial_controls=}")
        surrogate = Surrogate(
            bounds=(lb, ub),
            optimize_restarts=args.optimize_restarts,
            acquisition=args.acquisition,
        )

    controller = controls.ThrusterController(
        args.cal_file, propellant=args.gas, verbose=args.verbose
    )

    print("Starting surrogate control.")
    print(f"Control variables: {control_vars}")
    print(f"Bounds: {bounds}")
    print(f"Data sources: {args.data}")
    print(f"Steps: {args.num_steps}")
    print()

    print("Initial control points:")
    for i, c in enumerate(initial_controls):
        print(f"    {i + 1}: {c}")
    print()

    should_exit=False
    best_setpoint = None
    best_metric = np.inf
    max_no_improvement = 10
    no_improvement_timer = 0

    with labview.LabViewClient(host=args.host_ip, port=args.port, timeout=60) as client:
        for step in range(args.num_steps + len(initial_controls)):
            step_num = step + start_step + 1

            if step < len(initial_controls):
                c_current = initial_controls[step]
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

            setpoint = vector_to_setpoint(
                base_setpoint,
                control_vars,
                c_current,
            )

            print()
            print(f"Step {step_num}/{args.num_steps}")
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

            z_actual = metric_fn(data, setpoint)
            if z_actual < best_metric:
                best_setpoint = setpoint
                best_metric = z_actual
                no_improvement_timer = 0
            elif surrogate.is_trained:
                no_improvement_timer += 1
                print(f"No improvement on best. Timer = {no_improvement_timer}/{max_no_improvement}")
                if no_improvement_timer >= max_no_improvement:
                    break

            mean_current = data["dmm"]["current"]
            rms_pct = rms_amplitude_pct(data, setpoint)
            rms_raw = rms_amplitude_raw(data, setpoint)
            print(f"Mean: {mean_current:.3f} A, RMS Amplitude: {rms_raw:.3f} A ({rms_pct*100:.2f}%)")

            if args.objective == "efficiency":
                thrust = data["thrust"]
                efficiency = compute_efficiency(data, setpoint)
                print(f"Thrust: {thrust:.3f} mN, efficiency: {efficiency:.3f}")

            surrogate.update(c_current, z_actual)
            if surrogate.is_trained:
                if dim == 1:
                    fig, axs = plt.subplots(2,1, layout='constrained', figsize=(6,6))
                    surrogate.plot_1d_on_axis(axs[1])
                    x = np.linspace(lb[0], ub[0], 100)
                    if surrogate.acquisition == "ei":
                        ei = [surrogate.expected_improvement([_x]) for _x in x]
                        axs[0].plot(x, ei, color = 'red')
                        axs[0].set(title="Expected improvement", xticklabels = [], xlim=(lb, ub))
                    elif surrogate.acquisition == "eig":
                        ei = [surrogate.expected_information_gain([_x]) for _x in x]
                        axs[0].plot(x, ei, color = 'red')
                        axs[0].set(title="Expected information gain", xticklabels = [], xlim=(lb, ub))

                    fig.savefig("surrogate.png")
                    plt.close(fig)
                elif dim == 2:
                    metadata = {
                        "variable_name": control_vars,
                        "metric_name": args.objective,
                    }
                    plot_dir = Path(output_dir) / "plots"
                    os.makedirs(plot_dir, exist_ok=True)
                    plot_surrogate(surrogate, metadata, step_num, plot_dir)

            sample = {
                "step": {step_num},
                "z_actual": z_actual,
                "z_pred": z_pred,
                "control_vars": control_vars,
                "control_vector": c_current,
                "setpoint": setpoint.model_dump(),
                "data": data,
                "surrogate": surrogate.to_dict(),
            }


            out_file = output_dir / f"{args.prefix}_{step_num:03d}.pkl"
            with open(out_file, "wb") as fd:
                pickle.dump(sample, fd)

            print(f"z = {z_actual:.6g} (best = {best_metric:.6g})")
            print(f"Surrogate trained: {surrogate.is_trained}")
            print(f"Saved: {out_file}")

        if args.reset_at_end or best_setpoint is None:
            print("Resetting to base setpoint.")
            controller.control_to(base_setpoint, client)
        else:
            print(f"Setting to optimum: {best_setpoint}\n(metric = {best_metric:.6g})")
            controller.control_to(best_setpoint, client)

    print("Done.")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)