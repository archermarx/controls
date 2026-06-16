import argparse
import logging
from pathlib import Path

import lib.controls as controls
from lib.labview import LabViewClient

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("file", type=Path, help="The path to the command file to monitor")
parser.add_argument("--cal-file", "-c", type=Path, help="The path to the thruster calibration file")
parser.add_argument("--sleep-interval", type=float, default=0.25, help="How often, in seconds, to check for modifications to the command file")
parser.add_argument("--dummy", action="store_true")
parser.add_argument("--verbose", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()

    control_file = Path(args.file)

    client = controls.ThrusterController(
        args.cal_file,
        propellant="Kr",
        verbose=args.verbose,
    )

    with LabViewClient(dummy=args.dummy) as labview_client:
        if args.dummy:
            print("Using dummy labview client")

        while True:
            client.start_listening(
                labview_client,
                control_file,
                sleep_interval=args.sleep_interval, 
            )