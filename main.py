import argparse
import logging
import time
from pathlib import Path

from pydantic import BaseModel, ValidationError

from labview import LabViewClient, DeviceCommands, \
    AlicatControl, MagnaControl, LambdaControl,\
    set_alicat_control, set_lambda_control, set_magna_control

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("file", type=Path, help="The path to the command file to monitor")
parser.add_argument("--host-ip", type=str, default="169.254.144.78", help="The IP address of the LabView client")
parser.add_argument("--port", type=int, default=59704, help="The port of the labview client")
parser.add_argument("--sleep-interval", type=float, default=0.25, help="How often, in seconds, to check for modifications to the command file")

class ControlMetadata(BaseModel):
    counter: int = 0

class ControlPoint(BaseModel):
    anode_flow_rate_kg_s: float
    cathode_flow_fraction: float
    discharge_voltage_V: float
    magnet_current_inner_A: float
    magnet_current_outer_A: float

class ControlFile(BaseModel):
    metadata: ControlMetadata
    control: ControlPoint

def read_control_file(file, logger):
    try:
        with open(file, "r") as fd:
            return ControlFile.model_validate_json(fd.read())
    except ValidationError as e:
        raise

def build_lambda_commands(setpoint: ControlPoint):
    VOLTAGE_LIMIT=float('inf')
    inner = LambdaControl(
        label="inner",
        current_limit=setpoint.magnet_current_inner_A,
        voltage_limit=VOLTAGE_LIMIT,
        overvoltage_protection=VOLTAGE_LIMIT,
        enable=True,
    )
    outer = LambdaControl(
        label="outer",
        current_limit=setpoint.magnet_current_inner_A,
        voltage_limit=VOLTAGE_LIMIT,
        overvoltage_protection=VOLTAGE_LIMIT,
        enable=True,
    )
    return [inner, outer]

def build_magna_command(setpoint: ControlPoint):
    return MagnaControl(
        voltage_limit=setpoint.discharge_voltage_V,
        current_limit=40.0, # TODO: expose these as params
        overcurrent_trip=30.0,
        overvoltage_trip=1000.0,
        enable=True,
    )

def build_alicat_command(setpoint: ControlPoint):
    anode_flow_rate_mg_s = setpoint.anode_flow_rate_kg_s
    cathode_flow_rate_mg_s = anode_flow_rate_mg_s * setpoint.cathode_flow_fraction

    anode_control = AlicatControl(
        label="anode",
        setpoint=anode_flow_rate_mg_s,
        units="mg/s",
    )

    cathode_control = AlicatControl(
        label="cathode",
        setpoint=cathode_flow_rate_mg_s,
        units="mg/s",
    )

    return [anode_control, cathode_control]

def send_model_setpoints_to_labview(client: LabViewClient, setpoint: ControlPoint):
    alicat_commands = build_alicat_command(setpoint)
    lambda_commands = build_lambda_commands(setpoint)
    magna_commands = build_magna_command(setpoint)

    set_magna_control(client, magna_commands)
    set_alicat_control(client, alicat_commands)
    set_lambda_control(client, lambda_commands)

    return DeviceCommands(magna_commands, alicat_commands, lambda_commands)

def check_for_change(file: Path, counter, last_modified, contents, logger):
    if not file.exists():
        return counter, last_modified, {}, False

    modified_time = file.stat().st_mtime
    if modified_time <= last_modified:
        return counter, last_modified, {}, False

    try:
        contents = read_control_file(file, logger)
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
        contents = read_control_file(control_file, logger)
        counter = contents.metadata.counter
        last_modified = control_file.stat().st_mtime
    except (FileNotFoundError, ValidationError):
        counter = 0
        last_modified = 0
        contents = {}

    logger.info(f"{counter=}, {last_modified=}")

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
                logger.info("Sending to LabView")
                send_model_setpoints_to_labview(client, control)

            time.sleep(args.sleep_interval)