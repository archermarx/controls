import argparse
from pathlib import Path

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