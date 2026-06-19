import argparse
import time
from pathlib import Path
from typing import get_args
import os

import numpy as np
import lib.controls as controls
import lib.labview as labview
import lib.surrogate as surrogate

from lib.diffusion_control import DiffusionController
from lib.forward_model import ForwardModel
from lib.reverse_model import ReverseModel


def split_and_strip(text, dlm=","):
    return [x.strip() for x in text.split(dlm) if x.strip()]


parser = argparse.ArgumentParser()

parser.add_argument(
    "--cal-file", "-c", type=Path, help="The path to the thruster calibration file"
)
parser.add_argument("--setpoint", "-s", type=Path, required=True)
parser.add_argument("--control-vars", type=str, required=True)
parser.add_argument("--bounds", type=str, required=True)
parser.add_argument("--data", "-d", type=split_and_strip)
parser.add_argument("--num-steps", "-n", type=int, default=25)
parser.add_argument("--dwell-time", "-t", type=int, default=5)
parser.add_argument("--gas", "-g", type=str, choices=["Kr"], default="Kr")

parser.add_argument("--host-ip", type=str, default=labview.LABVIEW_IP)
parser.add_argument("--port", type=int, default=labview.LABVIEW_PORT)
parser.add_argument("--verbose", "-v", action="store_true")

parser.add_argument("--optimize-restarts", type=int, default=5)
parser.add_argument(
    "--acquisition", type=str, choices=["ei", "eig", "mean"], default="ei"
)

parser.add_argument("--output", "-o", type=Path, default=Path("."))
parser.add_argument("--metric", type=str)
parser.add_argument("--remote-dir", type=Path)
parser.add_argument("--max-no-improvement", type=int, default=10)
parser.add_argument("--num-reverse-samples", type=int, default=32)
parser.add_argument("--num-reverse-steps", type=int, default=512)
parser.add_argument("--diffusion-model-path", type=str)
parser.add_argument(
    "--surrogate-type", type=str, choices=get_args(surrogate.ModelType), default="KRG"
)
parser.add_argument("--control-file", type=str, default="")
parser.add_argument("--forward-thruster-config", type=str)
parser.add_argument("--dataset-dir", type=str)
parser.add_argument("--interactive", action="store_true")
parser.add_argument("--replay", type=str)
parser.add_argument("--replay-step", type=int, default=1)
parser.add_argument("--no-surrogate", action="store_true")


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


def main():
    args = parser.parse_args()

    metric_fn = controls.pick_metric(args.metric)
    control_vars = split_and_strip(args.control_vars)

    thruster_controller = controls.ThrusterController(
        args.cal_file,
        propellant=args.gas,
        verbose=args.verbose,
        control_to_file=args.control_file,
    )

    bounds = parse_bounds(args.bounds)
    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]

    kernel = "matern32" if args.surrogate_type == "KRG" else "squar_exp"
    acquisition = args.acquisition

    if args.no_surrogate:
        surr = None
    else:
        surr = surrogate.Surrogate(
            bounds=(lb, ub),
            model_type=args.surrogate_type,
            kernel=kernel,
            acquisition=acquisition,
        )

    forward_model = ForwardModel(
        case_config=args.forward_thruster_config,
        dataset_dir=args.dataset_dir,
        num_workers=args.num_reverse_samples,
        duration=2e-3,
    )

    reverse_config = {
        "observation": {
            "base_sim": args.dataset_dir,
            "stddev": 1.025,
        },
        "S_churn": 20.0,
        "S_tmin": 0.05,
        "S_tmax": 40,
        "rk_alpha": 1.0,
        "model": "ema",
    }

    reverse_model = ReverseModel(
        model=args.diffusion_model_path, config=reverse_config, sample_dir="samples"
    )

    base_setpoint = controls.read_setpoint(args.setpoint)

    c0 = {
        "discharge_voltage_v": base_setpoint.discharge_voltage_v,
        "anode_mass_flow_rate_kg_s": base_setpoint.anode_mass_flow_rate_kg_s,
        "magnetic_field_scale": base_setpoint.magnetic_field_scale,
        "cathode_coupling_voltage_v": 10.0,
    }

    diff_controller = DiffusionController(
        c0=c0,
        forward=forward_model,
        reverse=reverse_model,
        num_reverse_steps=args.num_reverse_steps,
        num_reverse_samples=args.num_reverse_samples,
        controller=thruster_controller,
        control_vars=control_vars,
        metric=metric_fn,
        control_lb=lb,
        control_ub=ub,
        surrogate=surr,
        Trust_relaxation=0.2,
        step_scale=0.33,
    )

    cs = []
    zs = []
    best_metric = np.inf
    best_metric_pt = c0

    data_args = {
        "delay": args.dwell_time,
        "sources": args.data,
        "num_thrust_points": 20,
    }

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    assert os.path.exists(args.diffusion_model_path)
    assert os.path.exists(args.dataset_dir)
    assert os.path.exists(args.forward_thruster_config)

    with labview.LabViewClient(dummy=(args.control_file != "")) as client:
        for i in range(args.num_steps):
            cs.append(diff_controller.control_point)
            c_next, z = diff_controller.step(client, **data_args)
            zs.append(z)

            if z < best_metric:
                best_metric = z
                best_metric_pt = cs[-1]

            diff_controller.save_to_file(Path(args.output) / "diffusion_log.pkl")

            print(f"======= Step {i + 1} / {args.num_steps} =======")
            print("Control point: ")
            for val, name in zip(
                diff_controller.dict_to_vec(cs[-1]),
                diff_controller.control_vars,
            ):
                print(f"    {name}: {val:.3g}")
            print(f"Metric: {diff_controller.zs[-1]}")
            print(f"Best metric: {best_metric} at {best_metric_pt}")
            print(f"Model trust: {diff_controller.model_trust}")
            print(f"Next point: {c_next}")

            if i < args.num_steps - 1:
                if args.interactive:
                    while input("Continue to next point? (y/n): ").casefold() != "y":
                        time.sleep(0.1)
                print()
        print()


if __name__ == "__main__":
    main()
