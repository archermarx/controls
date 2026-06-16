import argparse
import logging
import time
import pickle
from pathlib import Path

from pydantic import ValidationError

import lib.controls as controls
import lib.labview as labview
from lib.labview import LabViewClient

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("file", type=Path, help="The path to the command file to monitor")
parser.add_argument("--cal-file", "-c", type=Path, help="The path to the thruster calibration file")
parser.add_argument("--sleep-interval", type=float, default=0.25, help="How often, in seconds, to check for modifications to the command file")
parser.add_argument("--dwell-time", "-t", type=int, default=5, help="How long (in seconds) to dwell at each operating point before collecting data.")

if __name__ == "__main__":
    args = parser.parse_args()

    control_file = Path(args.file)
    data_file = Path(args.data_file)

    client = controls.ThrusterController(
        args.cal_file,
        propellant="Kr",
    )

    with LabViewClient(dummy=True) as labview_client:
        while True:
            client.start_listening(
                labview_client,
                control_file,
                sleep_interval=args.sleep_interval, 
            )