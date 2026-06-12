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
parser.add_argument("--host-ip", type=str, default=labview.LABVIEW_IP, help="The IP address of the LabVIEW client")
parser.add_argument("--port", type=int, default=labview.LABVIEW_PORT, help="The port of the LabVIEW client")
parser.add_argument("--sleep-interval", type=float, default=0.25, help="How often, in seconds, to check for modifications to the command file")
parser.add_argument("--verbose", "-v", action="store_true", help="Whether to print extra information, including raw byte strings sent to labview")
parser.add_argument("--data-file", "-d", type=Path, help="The file to which we write data received from LabVIEW. No data will be taken if this is empty")
parser.add_argument("--dwell-time", "-t", type=int, default=5, help="How long (in seconds) to dwell at each operating point before collecting data.")

def check_for_change(file: Path, counter, last_modified, contents, logger):
    if not file.exists():
        return counter, last_modified, {}, False

    modified_time = file.stat().st_mtime
    if modified_time <= last_modified:
        return counter, last_modified, {}, False

    try:
        contents = controls.read_control_file(file, logger)
    except (PermissionError, FileNotFoundError, ValidationError):
        return counter, last_modified, {}, False

    new_counter = contents.metadata.counter
    if new_counter > counter:
        return new_counter, modified_time, contents.control, True
    else:
        return counter, last_modified, None, False

if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    control_file = Path(args.file)

    try:
        contents = controls.read_control_file(control_file, logger)
        counter = contents.metadata.counter
        last_modified = control_file.stat().st_mtime
    except (FileNotFoundError, ValidationError):
        counter = 0
        last_modified = 0
        contents = {}

    logger.info(f"{counter=}, {last_modified=}")

    controller = controls.ThrusterController(args.cal_file, propellant = "Kr", verbose=False)

    with LabViewClient(host=args.host_ip, port=args.port) as client:
        while True:
            counter, last_modified, control, changed = check_for_change(control_file, counter, last_modified, contents, logger)
            if changed:
                assert isinstance(control, controls.ControlPoint)
                control_dict = control.model_dump()
                status_str = f"New setpoint received (counter={counter})"
                status_str = "\n" + status_str + "\n" + "-"*len(status_str) + "\n"
                for (k, v) in control_dict.items():
                    status_str += f"    {k}: {v}\n"
                
                logger.info(status_str)
                logger.info("Sending to LabView...")
                controller.control_to(control, client=client)

                # Taking data
                if args.data_file is not None:
                    data = controller.take_data(client=client, delay=args.dwell_time)

                    # Save data to file
                    with open(args.data_file, "wb") as fd:
                        pickle.dump(data, fd)

                    logger.info(f"Data saved to {args.data_file}.")

            time.sleep(args.sleep_interval)