from __future__ import annotations

import os
import time
from pydantic import BaseModel, ValidationError
from pathlib import Path
import json
import pickle

from dataclasses import asdict

import numpy as np
from typing import Literal, get_args

import lib.labview as labview
from lib.labview import (
    LabViewClient,
    MagnaControl,
    AlicatControl,
    LambdaControl,
)

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

ControlType = Literal[
    "no_action",
    "set_control",
    "receive_control",
    "take_data",
    "send_data",
]


class ControlMetadata(BaseModel):
    counter: int = 0
    type: ControlType = "no_action"


class ControlFile(BaseModel):
    metadata: ControlMetadata
    payload: dict


class ControlPoint(BaseModel):
    anode_mass_flow_rate_kg_s: float
    cathode_flow_fraction: float
    discharge_voltage_v: float
    magnetic_field_scale: float
    magnetic_field_scale_outer: float | None = None


def read_control_file(file):
    try:
        with open(file, "rb") as fd:
            contents = pickle.load(fd)
        return ControlFile.model_validate(contents)
    except ValidationError:
        raise


def check_for_change(file: Path | str, counter, last_modified):
    no_change_payload = (counter, None, last_modified, {}, False)
    file = Path(file)
    if not file.exists():
        return no_change_payload

    modified_time = file.stat().st_mtime
    if modified_time <= last_modified:
        return no_change_payload

    try:
        contents = read_control_file(file)
    except (PermissionError, FileNotFoundError, ValidationError):
        return no_change_payload

    new_counter = contents.metadata.counter
    if new_counter > counter:
        return (
            new_counter,
            contents.metadata.type,
            modified_time,
            contents.payload,
            True,
        )
    else:
        return no_change_payload


def time_str(s):
    if s >= 3600:
        h = s // 3600
        m = (s - 3600 * h) // 60
        s = s - 3600 * h - 60 * m
        return f"{h}h {m}m {s}s"
    elif s >= 60:
        m = s // 60
        s = s - 60 * m
        return f"{m}m {s}s"
    else:
        return f"{s} s"


def countdown(total, status_fn):
    line_len = len(status_fn(total))
    for t in range(total, 0, -1):
        print(" " * line_len, end="\r")
        print(status_fn(t), end="\r")
        time.sleep(1)
    print("\n")


def calibrate(val, cal):
    return val * cal[0] + cal[1]


def apply_limits(val, range):
    if val < range[0] or val > range[1]:
        raise ValueError(f"Value {val} exceeded range of {range}")
    return val


def calc_performance_metrics(data, setpoint: ControlPoint):
    dmm: dict = data["dmm"]
    mean_current_dmm = dmm["current"]
    discharge_current_trace = labview.OscopeReadings.from_dict(
        data["oscope"]["Anode Current"]
    )
    time, discharge_current_oscope = (
        discharge_current_trace.waveform.time_values(),
        discharge_current_trace.waveform.y_values(),
    )
    mean_current_oscope = np.mean(discharge_current_oscope)
    discharge_current_centered = discharge_current_oscope - mean_current_oscope
    rms_current = np.std(discharge_current_centered)
    discharge_current_signal = {
        "time": time,
        "current": discharge_current_oscope - mean_current_oscope + mean_current_dmm,
    }

    perf = {
        "discharge_current_A": mean_current_dmm,
        "discharge_current_rms_A": rms_current,
        "discharge_current_signal": discharge_current_signal,
    }

    if "thrust" in data:
        thrust_mN: float = data["thrust"]["thrust_mN"]
        thrust_N = thrust_mN / 1000
        current_A = dmm["current"]
        mdot = setpoint.anode_mass_flow_rate_kg_s
        vd = setpoint.discharge_voltage_v
        power_W = vd * current_A
        power_kW = power_W / 1000

        perf["anode_eff"] = 0.5 * thrust_N**2 / mdot / vd / current_A
        perf["thrust_to_power_mN_kW"] = thrust_mN / power_kW
        perf["isp_s"] = thrust_N / mdot / 9.81
        perf["thrust_N"] = thrust_N

    return perf


def rms_amplitude_raw(perf, *args, **kwargs):
    return perf["discharge_current_rms_A"]


def rms_amplitude_pct(perf, *args, **kwargs):
    return perf["discharge_current_rms_A"] / perf["discharge_current_A"]


def efficiency_obj(perf):
    return 1 - perf["anode_eff"]


def thrust_to_power_obj(perf, min_isp=2300):
    def smoothstep(edge0, edge1, x):
        t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
        return t * t * (3 - 2 * t)

    isp, t2p = perf["isp_s"], perf["thrust_to_power_mN_kW"]

    isp_penalty = smoothstep(1.025 * min_isp, 0.975 * min_isp, isp) * 100
    print(f"{isp=:.3f}, {isp_penalty=:.3f}")
    return -t2p + isp_penalty


def pick_metric(metric_str):
    if metric_str == "rms":
        metric_fn = rms_amplitude_pct
    elif metric_str == "efficiency":
        metric_fn = efficiency_obj
    elif metric_str == "thrust_to_power":
        metric_fn = thrust_to_power_obj
    else:
        raise ValueError(f"Invalid metric str {metric_str}")

    return metric_fn


def read_setpoint(path):
    with open(path, "rb") as fd:
        return ControlPoint.model_validate_json(fd.read())


def setpoint_to_vector(setpoint, control_vars):
    return np.array(
        [float(getattr(setpoint, name)) for name in control_vars],
        dtype=float,
    )


def vector_to_setpoint(base_setpoint, control_vars, c):
    setpoint = base_setpoint.model_copy(deep=True)
    for name, value in zip(control_vars, c):
        setattr(setpoint, name, float(value))
    return setpoint


class ThrusterController:
    def __init__(
        self,
        cal_file: str | Path,
        propellant: str = "Kr",
        verbose: bool = False,
        voltage_range: tuple[float, float] = (0, 800),
        flow_range: tuple[float, float] = (0, 800),
        control_to_file: str | Path = "",
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
        self.cathode_flow_range = flow_range
        # self.cathode_flow_range = (0.05 * flow_range[0], 0.1 * flow_range[1])

        # Oscope ranges
        self.oscope_time_width = 10e-3

        # If control_to_file is defined, we write controls to the given path instead of directly commanding the thruster
        self.control_to_file = control_to_file
        self.control_last_modified = 0.0
        self.control_counter = 0

        self.read_counter = lambda f: read_control_file(f).metadata.counter

        # Check current counter in control file
        if self.control_to_file != "" and os.path.exists(self.control_to_file):
            self.control_counter = self.read_counter(self.control_to_file)

    def wait_for_command(
        self, file, types: list[ControlType] | None = None, sleep_interval=0.1
    ):
        if types is None:
            allowed_types = get_args(ControlType)
        else:
            allowed_types = types

        if self.verbose:
            print(
                f"Waiting for command of type {allowed_types} (counter={self.control_counter})"
            )

        while True:
            counter, type, last_modified, contents, changed = check_for_change(
                file, self.control_counter, self.control_last_modified
            )

            if changed and type in allowed_types:
                if self.verbose:
                    print(f"Received command of type {type} (counter={counter})")
                break
            time.sleep(sleep_interval)

        self.control_counter = counter
        self.control_last_modified = last_modified
        return type, contents

    def send_command(self, file, type, payload: dict | None = None):
        assert file != ""

        self.control_counter += 1
        file_contents = ControlFile(
            metadata=ControlMetadata(counter=self.control_counter, type=type),
            payload=payload if payload else {},
        )

        with open(file, "wb") as fd:
            pickle.dump(file_contents.model_dump(), fd)

        self.control_last_modified = file.stat().st_mtime

        if self.verbose:
            print(f"Send command of type {type} (counter={self.control_counter})")

    def start_listening(
        self,
        client: LabViewClient,
        control_file: Path,
        sleep_interval: float = 0.1,
    ):
        # Create control file
        with open(control_file, "wb") as fd:
            metadata = ControlMetadata(counter=0, type="no_action")
            data = ControlFile(metadata=metadata, payload={})
            pickle.dump(data, file=fd)

        # Read current counter from control and data files
        self.control_counter = self.read_counter(control_file)

        print(f"Listening to file {control_file}  (counter={self.control_counter})")

        while True:
            type, payload = self.wait_for_command(
                control_file, sleep_interval=sleep_interval
            )

            if type == "set_control":
                setpoint = ControlPoint.model_validate(payload)
                self.control_to(setpoint, client)
                self.send_command(control_file, "receive_control")

            elif type == "take_data":
                data = self.take_data(client, **payload)
                self.send_command(control_file, "send_data", data)

    def kgs_to_sccm(self, kgs):
        return kgs * 1e6 * MGS_TO_SCCM[self.propellant]

    def set_flow(self, client, anode_sccm, cathode_sccm):
        # Set the flow controllers
        anode_flow_control = AlicatControl(
            label="anode",
            setpoint=apply_limits(
                calibrate(anode_sccm, self.cal["anode_flow"]), self.flow_range
            ),
        )
        cathode_flow_control = AlicatControl(
            label="cathode",
            setpoint=apply_limits(
                calibrate(cathode_sccm, self.cal["cathode_flow"]),
                self.cathode_flow_range,
            ),
        )
        alicat_control = [anode_flow_control, cathode_flow_control]
        labview.set_alicat_control(client, alicat_control)
        return alicat_control

    def set_magnets(self, client, inner_scale, outer_scale):
        # Set the magnet power supplies
        VOLTAGE_LIMIT = float("inf")

        inner_magnet = LambdaControl(
            label="inner",
            current_limit=calibrate(
                inner_scale * self.magnet_currents["inner"], self.cal["inner_magnet"]
            ),
            voltage_limit=VOLTAGE_LIMIT,
            overvoltage_protection=VOLTAGE_LIMIT,
            enable=True,
        )

        outer_magnet = LambdaControl(
            label="outer",
            current_limit=calibrate(
                outer_scale * self.magnet_currents["outer"], self.cal["outer_magnet"]
            ),
            voltage_limit=VOLTAGE_LIMIT,
            overvoltage_protection=VOLTAGE_LIMIT,
            enable=True,
        )

        lambda_control = [inner_magnet, outer_magnet]
        labview.set_lambda_control(client, lambda_control)
        return lambda_control

    def set_discharge_voltage(self, client, voltage):
        # Set the power supply
        magna_control = MagnaControl(
            voltage_limit=apply_limits(
                calibrate(voltage, self.cal["voltage"]), self.voltage_range
            ),
            overcurrent_trip=75,
            current_limit=100,
            overvoltage_trip=1000,
            enable=voltage > 0,
        )
        labview.set_magna_control(client, magna_control)
        return magna_control

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
            self.send_command(
                self.control_to_file, "set_control", self.setpoint.model_dump()
            )
            self.wait_for_command(self.control_to_file, types=["receive_control"])
            return

        anode_flow_rate_sccm = self.kgs_to_sccm(setpoint.anode_mass_flow_rate_kg_s)
        cathode_flow_rate_sccm = anode_flow_rate_sccm * setpoint.cathode_flow_fraction

        inner_scale = setpoint.magnetic_field_scale
        if setpoint.magnetic_field_scale_outer is None:
            outer_scale = inner_scale
        else:
            outer_scale = setpoint.magnetic_field_scale_outer

        if client.dummy:
            # Don't actually try and set controls if the client is set to dummy
            return

        if set_lambdas:
            self.set_magnets(client, inner_scale, outer_scale)

        if set_alicats:
            self.set_flow(client, anode_flow_rate_sccm, cathode_flow_rate_sccm)

        if set_magna:
            self.set_discharge_voltage(client, setpoint.discharge_voltage_v)

    def calc_thrust(self, s, s0):
        cal = self.cal["thrust_stand"]
        m = cal["slope"]
        b = cal["intercept"]
        return {"shunt": s, "thrust_mN": m * (s - s0) + b}

    def take_thrust_shutoff(
        self, client, num_avg_pts=10, settle_time=30, relight_time=5
    ):
        assert self.setpoint is not None

        dmm = labview.get_dmm_readings(client)
        current_before = dmm.current

        mdot = self.setpoint.anode_mass_flow_rate_kg_s
        vd = self.setpoint.discharge_voltage_v

        # Base setpoint
        on_pt = self.setpoint
        mdot_a_sccm = self.kgs_to_sccm(on_pt.anode_mass_flow_rate_kg_s)
        mdot_c_sccm = on_pt.cathode_flow_fraction * mdot_a_sccm

        # Get thrust before shutoff
        thrust_before = self.take_thrust(client, num_avg_pts=num_avg_pts)

        # Turn thruster off by cutting flow and discharge power
        shutoff_frac = 0.0
        self.set_discharge_voltage(client, shutoff_frac * vd)
        self.set_flow(
            client,
            anode_sccm=shutoff_frac * mdot_a_sccm,
            cathode_sccm=shutoff_frac * mdot_c_sccm,
        )

        countdown(
            settle_time, lambda t: "Settling at off point for " + time_str(t) + "."
        )

        # Get thrust after shutoff
        thrust_after = self.take_thrust(client, num_avg_pts, reset_null_shunt=True)

        s0 = thrust_after["shunt"]
        s = thrust_before["shunt"]
        thrust = self.calc_thrust(s, s0)
        print(
            f"Thrust measurement: {thrust['thrust_mN']:.2f} mN (shunt={thrust['shunt']})"
        )

        print("Relighting...")
        self.set_flow(client, anode_sccm=self.kgs_to_sccm(mdot), cathode_sccm=60)
        countdown(10, lambda t: "Flowing gas: " + time_str(t) + ".")
        print("Turning on voltage")
        self.set_discharge_voltage(client, vd)

        countdown(
            relight_time, lambda t: "Settling at on point for " + time_str(t) + "."
        )

        # Check if thruster actually lit
        dmm = labview.get_dmm_readings(client)
        current_after = dmm.current
        print(f"Current: before={current_before:.3f} A, after={current_after:.3f} A")
        if current_after < 0.5 * current_before:
            self.set_discharge_voltage(client, 0.0)
            time.sleep(1.0)
            raise ValueError(
                "Thruster failed to relight! Turning off voltage and aborting"
            )

        print("Returning to original setpoint")
        self.control_to(on_pt, client)

        return thrust

    def take_thrust(self, client, num_avg_pts=10, reset_null_shunt=False):
        config = labview.ThrustStandConfig(
            num_points=num_avg_pts,
            gains=labview.PIDGain(
                self.cal["thrust_stand"]["Kp"],
                self.cal["thrust_stand"]["Ki"],
                self.cal["thrust_stand"]["Kd"],
            ),
        )

        labview.set_thruststand_config(client, config)
        readings = labview.get_thruststand_readings(client)
        shunt = np.mean(readings.shunt)

        if reset_null_shunt:
            self.cal["thrust_stand"]["shunt_at_setpoint"] = shunt
            with open(self.cal_file, "w") as fd:
                print("Updated calibration")
                json.dump(
                    {
                        "magnet_currents_A": self.magnet_currents,
                        "calibration": self.cal,
                    },
                    fd,
                    indent=4,
                )

        s_mean = np.mean(shunt)
        s0 = self.cal["thrust_stand"]["shunt_at_setpoint"]
        return self.calc_thrust(s_mean, s0)

    def take_oscope(self, client: LabViewClient):
        # O-scope time base
        oscope_time_base = labview.OscopeTimeBase(
            range=self.oscope_time_width, position=0, reference=1
        )

        # Configure oscope to not collect waveforms so we can grab the averages and peak to peak amplitudes
        # The oscope has 8-bit depth so we want to ensure we get maximum resolution when we get waveforms
        # This requires that we rescale things on the fly
        # Note: the keys are hard-coded here. We shouldn't do this.
        assert self.setpoint is not None
        id = 50
        variable_settings = {
            "Anode Current": dict(offset=id / 2, range=id),
            "Cathode Current": dict(offset=id / 2, range=id),
            "Discharge Voltage": dict(
                offset=self.setpoint.discharge_voltage_v,
                range=self.setpoint.discharge_voltage_v,
            ),
            "C2G Voltage": dict(offset=-18, range=40),
        }
        channels = [
            labview.OscopeChannelConfig(
                k, range=v["range"], offset=v["offset"], collect_waveforms=False
            )
            for (k, v) in variable_settings.items()
        ]
        init_config = labview.OscopeConfig(
            time_base=oscope_time_base, channels=channels
        )
        labview.set_oscope_config(client, init_config)

        max_attempts = 3
        for attempt in range(max_attempts):
            # Read oscope to get p2p and average so we can rescale to a tighter window
            prelim_readings = labview.get_oscope_readings(client)

            # Configure oscope to collect waveforms
            # For each channel, we need to get the mean and p2p and use this to set the range
            waveform_channels = []
            for reading in prelim_readings:
                waveform_channels.append(
                    labview.OscopeChannelConfig(
                        label=reading.label,
                        range=(1.5 if reading.label != "C2G Voltage" else 2.5)
                        * reading.peak_to_peak,
                        offset=reading.average,
                        collect_waveforms=True,
                    )
                )

            waveform_config = labview.OscopeConfig(
                time_base=oscope_time_base, channels=waveform_channels
            )
            labview.set_oscope_config(client, waveform_config)
            oscope_readings = labview.get_oscope_readings(client)
            out = {r.label: asdict(r) for r in oscope_readings}

            # Reset ranges and turn off waveform collection
            labview.set_oscope_config(client, init_config)

            repeat = False
            for r in oscope_readings:
                if len(r.waveform.data) == 0:
                    print(
                        f"Warning: waveform not collected for channel {r.label}. Repeating (try {attempt + 1}/{max_attempts})."
                    )
                    repeat = True

            if not repeat:
                return out

        return None

    def take_data(
        self,
        client: LabViewClient,
        delay: int = 0,
        num_thrust_points=10,
        sources: list[str] | None = None,
    ):
        # Pause according to prescribed delay
        if delay > 0:
            countdown(delay, lambda t: "Waiting to take data: " + time_str(t) + ".")
            print("Taking data")

        if self.control_to_file != "":
            data_args = dict(
                delay=0, num_thrust_points=num_thrust_points, sources=sources
            )
            self.send_command(self.control_to_file, "take_data", data_args)
            _, data = self.wait_for_command(self.control_to_file, types=["send_data"])
            return data

        if not sources:
            data_sources = set(
                ["dmm", "magna", "alicat", "lambda", "oscope", "thruststand"]
            )
        else:
            data_sources = set(sources)

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

