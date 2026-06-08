import time
from pydantic import BaseModel, ValidationError

import lib.labview as labview
from lib.labview import LabViewClient, MagnaControl, AlicatControl, LambdaControl, \
                    DeviceCommands

# Conversion factors from mg/s to SCCM for noble gas propellants
MGS_TO_SCCM = {
    "Xe": 11.18,
    "Kr": 17.25,
    "Ar": 36.75,
}

# Approximate mass flow rates in mgs for 15 A for noble gas propellants.
# Used to set approximate current limits
# Xe and Kr data from Su et al. 2024. Argon number approximate.
CURRENT_PER_FLOW = {
    "Xe": 15 / 14.8,
    "Kr": 15 / 11.8,
    "Ar": 15 / 7.0,
}

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

def time_str(s):
    if s >= 3600:
        h = s // 3600
        m = (s - 3600*h) // 60
        s = s - 3600*h - 60*m
        return f"{h}h {m}m {s}s"
    elif s >= 60:
        m = s // 60
        s = s - 60*m
        return f"{m}m {s}s"
    else:
        return f"{s} s"

def status_str(t):
    return f"Waiting to take data. Time remaining: " + time_str(t) + "."

def calibrate(val, cal):
    return val * cal[0] + cal[1]

class ThrusterController:
    def __init__(self, propellant: str = "Kr", verbose: bool = False):
        self.setpoint = None
        self.verbose = verbose
        self.propellant = propellant

        # Calibration factors: (slope, intercept)
        self.voltage_cal      = (1.01888,  0.544818)
        self.inner_magnet_cal = (1.00808, -0.002797)
        self.outer_magnet_cal = (1.00483, -0.013443)
        self.anode_flow_cal   = (1.002283, 0.554064)
        self.cathode_flow_cal = (1.0, 0.0)

    def control_to(self, setpoint: ControlPoint, client: LabViewClient):
        if self.setpoint is None:
            self.setpoint = setpoint

        anode_flow_rate_mg_s = setpoint.anode_flow_rate_kg_s * 1e6
        anode_flow_rate_sccm = anode_flow_rate_mg_s * MGS_TO_SCCM[self.propellant]
        cathode_flow_rate_sccm = anode_flow_rate_sccm * setpoint.cathode_flow_fraction

        # Calculate the expected current so we can set appropriate current limits
        expected_current = anode_flow_rate_mg_s * CURRENT_PER_FLOW[self.propellant]
        overcurrent = 3 * expected_current
        current_limit = 1.25 * overcurrent
        overvoltage = 1000

        # Set the power supply
        magna_control = MagnaControl(
            voltage_limit=calibrate(setpoint.discharge_voltage_V, self.voltage_cal),
            current_limit=current_limit,
            overcurrent_trip=overcurrent,
            overvoltage_trip=overvoltage,
            enable=True,
        )

        # Set the flow controllers
        anode_flow_control = AlicatControl(
            label="anode",
            setpoint=calibrate(anode_flow_rate_sccm, self.anode_flow_cal),
            units="sccm"
        )
        cathode_flow_control = AlicatControl(
            label="cathode",
            setpoint=calibrate(cathode_flow_rate_sccm, self.cathode_flow_cal),
            units="sccm"
        )
        alicat_control = [anode_flow_control, cathode_flow_control]

        # Set the magnet power supplies
        VOLTAGE_LIMIT=float('inf')
        lambda_control = [
            LambdaControl(
                label=label,
                current_limit=calibrate(current, cal),
                voltage_limit=VOLTAGE_LIMIT,
                overvoltage_protection=VOLTAGE_LIMIT,
                enable=True
            )
            for label, current, cal in zip(
                ["inner", "outer"],
                [setpoint.magnet_current_inner_A, setpoint.magnet_current_outer_A],
                [self.inner_magnet_cal, self.outer_magnet_cal],
            )
        ]

        labview.set_magna_control(client, magna_control, self.verbose)
        labview.set_alicat_control(client, alicat_control, self.verbose)
        labview.set_lambda_control(client, lambda_control, self.verbose)
        return DeviceCommands(magna_control, alicat_control, lambda_control)

    def take_data(self, client: LabViewClient, delay: int = 0, sources: list[str] | None = None):
        if sources is None:
            sources = ["dmm", "magna", "alicat", "lambda", "oscope"]

        if len(sources) == 0:
            raise ValueError("Sources array must not be empty!")

        # Pause according to prescribed delay
        if delay > 0:
            print()
            line_len = len(status_str(delay))
            for t in range(delay, 0, -1):
                print(" "*line_len, end="\r")
                print(status_str(t), end="\r")
                time.sleep(1)
        print("\nTaking data...")

        out = {}
        if "dmm" in sources:
            out["dmm"] = labview.get_dmm_readings(client)
        if "magna" in sources:
            out["magna"] = labview.get_magna_readings(client)
        if "alicat" in sources:
            alicat_readings = labview.get_alicat_readings(client)
            out["alicat"] = {r.label: r for r in alicat_readings}
        if "lambda" in sources:
            lambda_readings = labview.get_lambda_readings(client)
            out["lambda"] = {r.label: r for r in lambda_readings}
        if "oscope" in sources:
            # Configure oscope to collect waveforms
            # Note: we here assume 4 channels and assign empty labels
            # In future it may be good to check this and pre-assign things
            configs = [labview.OscopeConfig("", collect_waveforms=True) for _ in range(4)]
            labview.set_oscope_config(configs)

            oscope_readings = labview.get_oscope_readings(client)
            out["oscope"] = {r.label: r for r in oscope_readings}

        return out
        