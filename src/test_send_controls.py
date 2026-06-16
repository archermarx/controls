import argparse
import logging
from pathlib import Path

import lib.controls as controls
import lib.labview as labview
from lib.labview import LabViewClient

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("file", type=Path, help="The path to the command file to monitor")
parser.add_argument(
    "--cal-file", "-c", type=Path, help="The path to the thruster calibration file"
)
parser.add_argument(
    "--sleep-interval",
    type=float,
    default=0.25,
    help="How often, in seconds, to check for modifications to the command file",
)
parser.add_argument(
    "--dwell-time",
    "-t",
    type=int,
    default=5,
    help="How long (in seconds) to dwell at each operating point before collecting data.",
)

voltages = [1.0, 2.0, 3.0, 4.0, 5.0]

if __name__ == "__main__":
    args = parser.parse_args()

    control_file = Path(args.file)

    server = controls.ThrusterController(
        args.cal_file,
        propellant="Kr",
        control_to_file=control_file,
        verbose=True,
    )

    with LabViewClient(dummy=True) as labview_client:
        for voltage in voltages:
            setpoint = controls.ControlPoint(
                anode_mass_flow_rate_kg_s=0.0,
                magnetic_field_scale=0.0,
                cathode_flow_fraction=0.0,
                discharge_voltage_v=voltage,
            )
            print(f"Controlling to setpoint: {setpoint}")
            server.control_to(setpoint, client=labview_client)
            data = server.take_data(client=labview_client, num_thrust_points=25, delay=5)
            print(f"Got data: keys = {list(data.keys())}")

            oscope = labview.OscopeReadings.from_dict(data["oscope"]["Anode Current"])
            waveform = oscope.waveform
            t = waveform.time_values()
            print(f"Interval: {t[-1] - t[0]:.3g} s")