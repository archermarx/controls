from __future__ import annotations

import os
import time
from pydantic import BaseModel, ValidationError
from pathlib import Path
import json

from dataclasses import asdict

import numpy as np

import lib.labview as labview
from lib.labview import LabViewClient, MagnaControl, AlicatControl, LambdaControl, DeviceCommands

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
    magnetic_field_scale_outer: float | None = None

class ControlFile(BaseModel):
    metadata: ControlMetadata
    control: ControlPoint

class DataFile(BaseModel):
    metadata: ControlMetadata
    data: dict

def read_control_file(file, cls):
    try:
        with open(file, "r") as fd:
            return cls.model_validate_json(fd.read())
    except ValidationError as e:
        raise

def check_for_change(file: Path | str, cls, counter, last_modified):
    file = Path(file)
    if not file.exists():
        return counter, last_modified, {}, False

    modified_time = file.stat().st_mtime
    if modified_time <= last_modified:
        return counter, last_modified, {}, False

    try:
        contents = read_control_file(file, cls)
    except (PermissionError, FileNotFoundError, ValidationError):
        return counter, last_modified, {}, False

    new_counter = contents.metadata.counter
    if new_counter > counter:
        return new_counter, modified_time, contents.control if cls == ControlFile else contents.data, True
    else:
        return counter, last_modified, {}, False

def wait_for_change(file, cls, counter, last_modified, sleep_interval=0.1):
    while True:
        counter, last_modified, contents, changed = check_for_change(file, cls, counter, last_modified)
        if changed:
            break
        time.sleep(sleep_interval)

    return counter, last_modified, contents

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
            voltage_range: tuple[float, float] = (0, 800),
            flow_range: tuple[float, float] = (0, 800),
            control_to_file: str | Path = "",
            data_from_file: str | Path = "",
        ):
        self.setpoint = None
        self.verbose = verbose
        self.propellant = propellant

        self.cal_file = cal_file

        # Read calibration file
        with open(self.cal_file, "rb") as fd:
            cal_dict = json.load(fd)

        self.cal = cal_dict["calibration"]
        self.magnet_currents = cal_dict["magnet_currents_A"]

        # Limits
        self.voltage_range = voltage_range
        self.flow_range = flow_range
        self.cathode_flow_range = (0.05 * flow_range[0], 0.1 * flow_range[1])

        # Oscope ranges
        self.oscope_time_width = 10e-3

        # If control_to_file is defined, we write controls to the given path instead of directly commanding the thruster
        # If data_from_file is defined, we wait to read data from the given file instead of directly taking it
        self.control_to_file = control_to_file
        self.data_from_file = data_from_file
        self.control_last_modified = 0.0
        self.data_last_modified = 0.0
        self.control_counter = 0
        self.data_counter = 0

        # Check current counter in control file
        if self.control_to_file is not None and os.path.exists(self.control_to_file):
            contents = read_control_file(self.control_to_file, ControlFile)
            self.control_counter = contents.metadata.counter + 1

        # Check current counter in data file
        if self.data_from_file is not None and os.path.exists(self.data_from_file):
            contents = read_control_file(self.data_from_file, DataFile)
            self.data_counter = contents.metadata.counter

    def write_control_file(self, setpoint: ControlPoint):
        assert self.control_to_file != ""

        file_contents = ControlFile(
            metadata = ControlMetadata(counter = self.control_counter),
            control = setpoint,
        )
        with open(self.control_to_file, "w") as fd:
            json.dump(file_contents.model_dump(), fd, indent=4)

        self.control_counter += 1

    def read_data_file(self):
        assert self.data_from_file != ""

        counter, last_modified, data = wait_for_change(
            self.data_from_file, DataFile, self.data_counter, self.data_last_modified
        )
        self.data_counter = counter
        self.data_last_modified = last_modified
        return data

    def start_listening(self,
            client: LabViewClient,
            control_file: Path,
            data_file: Path,
            sleep_interval: float = 0.1,
            **data_args
        ):

        print(f"Listening to file {control_file}")

        # Read current counter and last modified from control file
        contents = read_control_file(control_file, ControlFile)
        self.control_counter = contents.metadata.counter

        while True:
            self.control_counter, self.control_last_modified, setpoint = wait_for_change(
                control_file, ControlFile, self.control_counter, self.control_last_modified,
                sleep_interval=sleep_interval,
            )
            print(f"Received new control point: {setpoint}")
            assert isinstance(setpoint, ControlPoint)
            self.control_to(setpoint, client)

            data = self.take_data(client, **data_args)
            data_file_contents = DataFile(metadata=ControlMetadata(counter=self.data_counter), data=data)
            self.data_counter += 1
            with open(data_file, "w") as fd:
                json.dump(data_file_contents.model_dump(), fd, indent=4)
            
            print(f"Data saved to {data_file}")

    def control_to(
            self,
            setpoint: ControlPoint,
            client: LabViewClient,
            set_lambdas: bool = True,
            set_alicats: bool = True,
            set_magna: bool = True,
            ):

        self.setpoint = setpoint

        if self.control_to_file != "":
            self.write_control_file(self.setpoint)
            return

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
        )
        cathode_flow_control = AlicatControl(
            label="cathode",
            setpoint=apply_limits(
                calibrate(cathode_flow_rate_sccm, self.cal["cathode_flow"]),
                self.cathode_flow_range,
            ),
        )
        alicat_control = [anode_flow_control, cathode_flow_control]

        # Set the magnet power supplies
        VOLTAGE_LIMIT=float('inf')
        inner_scale = setpoint.magnetic_field_scale
        if setpoint.magnetic_field_scale_outer is None:
            outer_scale = inner_scale
        else:
            outer_scale = setpoint.magnetic_field_scale_outer

        inner_magnet = LambdaControl(
            label="inner",
            current_limit=calibrate(inner_scale * self.magnet_currents["inner"], self.cal["inner_magnet"]),
            voltage_limit=VOLTAGE_LIMIT,
            overvoltage_protection=VOLTAGE_LIMIT,
            enable=True
        )

        outer_magnet = LambdaControl(
            label="outer",
            current_limit=calibrate(outer_scale * self.magnet_currents["outer"], self.cal["outer_magnet"]),
            voltage_limit=VOLTAGE_LIMIT,
            overvoltage_protection=VOLTAGE_LIMIT,
            enable=True
        )
        lambda_control = [inner_magnet, outer_magnet]

        if client.dummy:
            # Don't actually try and set controls if the client is set to dummy
            return

        if set_magna:
            labview.set_magna_control(client, magna_control)

        if set_alicats:
            labview.set_alicat_control(client, alicat_control)

        if set_lambdas:
            print("Setting lambdas")
            labview.set_lambda_control(client, lambda_control)
            
        return DeviceCommands(magna_control, alicat_control, lambda_control)

    def take_thrust(self, client, num_avg_pts=10, reset_calibration=False):
        config = labview.ThrustStandConfig(
            num_points = num_avg_pts,
            gains = labview.PIDGain(
                self.cal["thrust_stand"]["Kp"],
                self.cal["thrust_stand"]["Ki"],
                self.cal["thrust_stand"]["Kd"],
            )
        )

        labview.set_thruststand_config(client, config)
        readings = labview.get_thruststand_readings(client)
        shunt = np.mean(readings.shunt)

        if reset_calibration:
            self.cal["thrust_stand"]["shunt_at_setpoint"] = shunt
            with open(self.cal_file, "w") as fd:
                print("Updated calibration")
                json.dump({"magnet_currents_A": self.magnet_currents, "calibration": self.cal}, fd, indent=4)

        cal = self.cal["thrust_stand"]
        s_mean = np.mean(shunt)
        s0 = cal["shunt_at_setpoint"]
        m = cal["slope"]
        b = cal["intercept"]
        return {"shunt": s_mean, "thrust_N": m * (s_mean - s0) + b}

    def take_oscope(self, client: LabViewClient):
        # O-scope time base
        oscope_time_base = labview.OscopeTimeBase(range=self.oscope_time_width, position=0, reference=1)

        # Configure oscope to not collect waveforms so we can grab the averages and peak to peak amplitudes
        # The oscope has 8-bit depth so we want to ensure we get maximum resolution when we get waveforms
        # This requires that we rescale things on the fly
        # Note: the keys are hard-coded here. We shouldn't do this.
        vd = 300.0 if self.setpoint is None else self.setpoint.discharge_voltage_v
        self.current_limit = 40
        variable_settings = {
            "Anode Current": dict(offset=self.current_limit/2, range=self.current_limit),
            "Cathode Current": dict(offset=self.current_limit/2, range=self.current_limit),
            "Discharge Voltage": dict(offset=vd, range=vd),
            "C2G Voltage": dict(offset=-18, range=40),
        }
        channels = [
            labview.OscopeChannelConfig(k, range=v["range"], offset=v["offset"], collect_waveforms=False)
            for (k, v) in variable_settings.items()
        ]
        init_config = labview.OscopeConfig(time_base=oscope_time_base, channels=channels)
        labview.set_oscope_config(client, init_config)

        max_attempts = 3
        for attempt in range(max_attempts):
            # Read oscope to get p2p and average so we can rescale to a tighter window
            prelim_readings = labview.get_oscope_readings(client)

            # Configure oscope to collect waveforms
            # For each channel, we need to get the mean and p2p and use this to set the range
            waveform_channels = []
            for reading in prelim_readings:
                waveform_channels.append(labview.OscopeChannelConfig(
                    label=reading.label,
                    range=(1.5 if reading.label != "C2G Voltage" else 2.5) * reading.peak_to_peak,
                    offset=reading.average,
                    collect_waveforms=True,
                ))

            waveform_config = labview.OscopeConfig(time_base=oscope_time_base, channels=waveform_channels)
            labview.set_oscope_config(client, waveform_config)
            oscope_readings = labview.get_oscope_readings(client)
            out = {r.label: asdict(r) for r in oscope_readings}

            # Reset ranges and turn off waveform collection
            labview.set_oscope_config(client, init_config)

            repeat = False
            for r in oscope_readings:
                if len(r.waveform.data) == 0:
                    print(f"Warning: waveform not collected for channel {r.label}. Repeating (try {attempt+1}/{max_attempts}).")
                    repeat = True
            
            if not repeat:
                return out

        return None

    def take_data(self, client: LabViewClient, delay: int = 0, num_thrust_points=10, sources: list[str] | None = None):
        assert self.setpoint is not None

        if self.data_from_file != "":
            return self.read_data_file()
        
        if not sources: 
            data_sources = set(["dmm", "magna", "alicat", "lambda", "oscope", "thruststand"])
        else:
            data_sources = set(sources)

        # Pause according to prescribed delay
        if delay > 0:
            line_len = len(status_str(delay))
            for t in range(delay, 0, -1):
                print(" "*line_len, end="\r")
                print(status_str(t), end="\r")
                time.sleep(1)
        print("\nTaking data...")

        out = {}
        if client.dummy:
            return out

        if "dmm" in data_sources:
            out["dmm"] = asdict(labview.get_dmm_readings(client))
        if "magna" in data_sources:
            out["magna"] = asdict(labview.get_magna_readings(client))
        if "alicat" in data_sources:
            alicat_readings = labview.get_alicat_readings(client)
            out["alicat"] = {r.label: asdict(r) for r in alicat_readings}
        if "lambda" in data_sources:
            lambda_readings = labview.get_lambda_readings(client)
            out["lambda"] = {r.label: asdict(r) for r in lambda_readings}
        if "thruststand" in data_sources:
            out["thrust"] = self.take_thrust(client, num_avg_pts=num_thrust_points)
        if "oscope" in data_sources:
            out["oscope"] = self.take_oscope(client)

        return out
        