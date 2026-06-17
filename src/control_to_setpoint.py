import argparse
from pathlib import Path
import numpy as np
import time

import lib.controls as controls
import lib.labview as labview

parser = argparse.ArgumentParser()
parser.add_argument("setpoint_file", type=Path)
parser.add_argument("--cal-file", "-c", type=Path)
parser.add_argument("--gas", "-g", type=str, default="Kr")
args = parser.parse_args()

with open(args.setpoint_file, "rb") as fd:
    setpoint = controls.ControlPoint.model_validate_json(fd.read())

controller = controls.ThrusterController(args.cal_file, propellant=args.gas)

with labview.LabViewClient() as client:
    controller.control_to(setpoint, client)

    # currents = []
    # data = controller.take_data(client, delay=10, sources=["dmm"])
    # currents.append(data["dmm"]["current"])

    # for i in range(50):
    #     data = controller.take_data(client, delay=0, sources=["dmm"])
    #     currents.append(data["dmm"]["current"])
    #     print(f"Current: {currents[-1]:.3f} A")
    #     time.sleep(0.1)

    # mean_current = np.mean(currents)
    # std_current = np.std(currents)

    #print(f"Final current: {mean_current:.3f} +/- {2*std_current:.3f} A")

    # thrust = controller.take_thrust_shutoff(client, num_avg_pts=10, settle_time=30, relight_time=5)
    # shunt = float(thrust["shunt"])
    # thrust_mN = float(thrust["thrust_mN"])
    # print(f"{shunt=}, {thrust_mN=}")