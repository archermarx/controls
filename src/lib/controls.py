from __future__ import annotations

import time
from pydantic import BaseModel, ValidationError
from pathlib import Path
import json

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
    anode_mass_flow_rate_kg_s: float
    cathode_flow_fraction: float
    discharge_voltage_v: float
    magnetic_field_scale: float

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

def apply_limits(val, range):
    if val < range[0] or val > range[1]:
        raise ValueError(f"Value {val} exceeded range of {range}")
    return val


class ThrusterController:
    def __init__(
            self, cal_file: str | Path, 
            propellant: str = "Kr",
            verbose: bool = False,
            voltage_range: tuple[float, float] = (200, 800),
            flow_range: tuple[float, float] = (100, 800),
        ):
        self.setpoint = None
        self.verbose = verbose
        self.propellant = propellant

        # Read calibration file
        with open(cal_file, "rb") as fd:
            cal_dict = json.load(fd)

        self.cal = cal_dict["calibration"]
        self.magnet_currents = cal_dict["magnet_currents_A"]

        # Limits
        self.voltage_range = voltage_range
        self.flow_range = flow_range
        self.cathode_flow_range = (0.05 * flow_range[0], 0.1 * flow_range[1])
    
    def control_to(
            self,
            setpoint: ControlPoint,
            client: LabViewClient,
            set_lambdas: bool = True,
            set_alicats: bool = True,
            set_magna: bool = True,
            ):
        self.setpoint = setpoint

        anode_flow_rate_mg_s = setpoint.anode_mass_flow_rate_kg_s * 1e6
        anode_flow_rate_sccm = anode_flow_rate_mg_s * MGS_TO_SCCM[self.propellant]
        cathode_flow_rate_sccm = anode_flow_rate_sccm * setpoint.cathode_flow_fraction

        MAX_CURRENT = 100

        # Calculate the expected current so we can set appropriate current limits
        expected_current = anode_flow_rate_mg_s * CURRENT_PER_FLOW[self.propellant]
        overcurrent = min(3 * expected_current, MAX_CURRENT)
        current_limit = 1.25 * overcurrent
        overvoltage = 1000
        self.current_limit = current_limit

        # Set the power supply
        magna_control = MagnaControl(
            voltage_limit=apply_limits(
                calibrate(setpoint.discharge_voltage_v, self.cal["voltage"]),
                self.voltage_range
            ),
            overcurrent_trip=75,
            current_limit=100,
            overvoltage_trip=overvoltage,
            enable=True,
        )

        # Set the flow controllers
        anode_flow_control = AlicatControl(
            label="anode",
            setpoint=apply_limits(
                calibrate(anode_flow_rate_sccm, self.cal["anode_flow"]),
                self.flow_range
            ),
            units="sccm"
        )
        cathode_flow_control = AlicatControl(
            label="cathode",
            setpoint=apply_limits(
                calibrate(cathode_flow_rate_sccm, self.cal["cathode_flow"]),
                self.cathode_flow_range,
            ),
            units="sccm"
        )
        alicat_control = [anode_flow_control, cathode_flow_control]

        # Set the magnet power supplies
        VOLTAGE_LIMIT=float('inf')
        lambda_control = [
            LambdaControl(
                label=label,
                current_limit=calibrate(setpoint.magnetic_field_scale * self.magnet_currents[label], cal),
                voltage_limit=VOLTAGE_LIMIT,
                overvoltage_protection=VOLTAGE_LIMIT,
                enable=True
            )
            for label, cal in zip(
                ["inner", "outer"],
                [self.cal["inner_magnet"], self.cal["outer_magnet"]],
            )
        ]

        if set_magna:
            labview.set_magna_control(client, magna_control, self.verbose)

        if set_alicats:
            labview.set_alicat_control(client, alicat_control, self.verbose)

        if set_lambdas:
            labview.set_lambda_control(client, lambda_control, self.verbose)
            
        return DeviceCommands(magna_control, alicat_control, lambda_control)

    def take_data(self, client: LabViewClient, delay: int = 0, sources: list[str] | None = None):
        assert self.setpoint is not None
        
        if not sources: 
            data_sources = set(["dmm", "magna", "alicat", "lambda", "oscope", "thruststand"])
        else:
            data_sources = set(sources)

        # Configure oscope to not collect waveforms so we can grab the averages and peak to peak amplitudes
        # The oscope has 8-bit depth so we want to ensure we get maximum resolution when we get waveforms
        # This requires that we rescale things on the fly
        # Note: the keys are hard-coded here. We shouldn't do this.
        if "oscope" in data_sources:
            variable_settings = {
                "Anode Current": dict(offset=self.current_limit/2, range=self.current_limit),
                "Cathode Current": dict(offset=self.current_limit/2, range=self.current_limit),
                "Discharge Voltage": dict(offset=self.setpoint.discharge_voltage_v, range=250),
                "C2G Voltage": dict(offset=-18, range=40),
            }
            init_configs = [
                labview.OscopeConfig(k, range=v["range"], offset=v["offset"], collect_waveforms=False)
                for (k, v) in variable_settings.items()
            ]
            labview.set_oscope_config(client, init_configs)

        # Pause according to prescribed delay
        if delay > 0:
            line_len = len(status_str(delay))
            for t in range(delay, 0, -1):
                print(" "*line_len, end="\r")
                print(status_str(t), end="\r")
                time.sleep(1)
        print("\nTaking data...")

        out = {}
        if "thruststand" in data_sources:
            out["thruststand"] = labview.get_thruststand_readings(client)
        if "dmm" in data_sources:
            out["dmm"] = labview.get_dmm_readings(client)
        if "magna" in data_sources:
            out["magna"] = labview.get_magna_readings(client)
        if "alicat" in data_sources:
            alicat_readings = labview.get_alicat_readings(client)
            out["alicat"] = {r.label: r for r in alicat_readings}
        if "lambda" in data_sources:
            lambda_readings = labview.get_lambda_readings(client)
            out["lambda"] = {r.label: r for r in lambda_readings}
        if "oscope" in data_sources:
            max_attempts = 3
            for attempt in range(max_attempts):
                # Read oscope to get p2p and average so we can rescale to a tighter window
                prelim_readings = labview.get_oscope_readings(client)

                # Configure oscope to collect waveforms
                # For each channel, we need to get the mean and p2p and use this to set the range
                waveform_configs = []
                for reading in prelim_readings:
                    waveform_configs.append(labview.OscopeConfig(
                        label=reading.label,
                        range=(1.5 if reading.label != "C2G Voltage" else 2.5) * reading.peak_to_peak,
                        offset=reading.average,
                        collect_waveforms=True,
                    ))

                labview.set_oscope_config(client, waveform_configs)
                oscope_readings = labview.get_oscope_readings(client)
                out["oscope"] = {r.label: r for r in oscope_readings}

                # Reset ranges and turn off waveform collection
                labview.set_oscope_config(client, init_configs)

                repeat = False
                for r in oscope_readings:
                    if len(r.waveform.data) == 0:
                        print(f"Warning: waveform not collected for channel {r.label}. Repeating (try {attempt+1}/{max_attempts}).")
                        repeat = True
                
                if not repeat:
                    break

        return out
        