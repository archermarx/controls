import argparse
import logging
import time
import pickle
from pathlib import Path

from pydantic import ValidationError

import controls
import labview
from labview import LabViewClient

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("file", type=Path, help="The path to the command file to monitor")
parser.add_argument("--host-ip", type=str, default="169.254.144.78", help="The IP address of the LabVIEW client")
parser.add_argument("--port", type=int, default=59704, help="The port of the LabVIEW client")
parser.add_argument("--sleep-interval", type=float, default=0.25, help="How often, in seconds, to check for modifications to the command file")
parser.add_argument("--verbose", "-v", action="store_true", help="Whether to print extra information, including raw byte strings sent to labview")
parser.add_argument("--data-file", "-d", type=Path, help="The file to which we write data received from LabVIEW. No data will be taken if this is empty")
parser.add_argument("--data-wait-time", "-t", type=int, default=2, help="The time (in seconds) we wait to take data after adjusting the setpoint.")

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
        return counter, last_modified, {}, False

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

    controller = controls.ThrusterController("Kr", verbose=False)

    with LabViewClient(host=args.host_ip, port=args.port) as client:
        while True:
            counter, last_modified, control, changed = check_for_change(control_file, counter, last_modified, contents, logger)
            if changed:
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
                    print()
                    for t in range(args.data_wait_time, 0, -1):
                        time_str = f"{t} s"
                        print("\r", end="")
                        logger.info(f"Waiting to take data. Time remaining: " + time_str)
                        time.sleep(1)
                    logger.info("Taking data...")

                    data = controller.take_data(client=client)

                    # Save data to file
                    with open(args.data_file, "w") as fd:
                        pickle.dump(data, fd)

            time.sleep(args.sleep_interval)