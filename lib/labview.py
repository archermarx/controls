from __future__ import annotations

import numpy as np

from dataclasses import dataclass, asdict
from typing import Any

import socket, struct, time

# GLOBAL VARIABLES

# ---------------------------------------------------------------------------
# Communication variables

# TCP Client Configuration
LABVIEW_IP = '169.254.144.78'
LABVIEW_PORT = 59704                                 # Set this to actual TCP port for receiving data from LabView
SOCKET_TIMEOUT = 7.0

# Keep False unless LabVIEW expects a flattened empty string argument for ""
SEND_EMPTY_ARG = False

HEADER_FMT = '>bi'                                  # byte command_id, signed int payload_length
HEADER_LENGTH = struct.calcsize(HEADER_FMT)

# Command Constants from API_Short.xlsx
CMD_MAGNA_GET_READINGS = 0x12                       # Command 18
CMD_MAGNA_SET_CONTROL = 0x13                        # Command 19

CMD_ALICAT_GET_READINGS = 0x1B                      # Command 27
CMD_ALICAT_SET_CONTROL = 0x1C                       # Command 28

CMD_LAMBDA_GET_READINGS = 0x23                      # Command 35
CMD_LAMBDA_SET_CONTROL = 0x24                       # Command 36

CMD_OSCOPE_SET_CONFIG = 0x2A                        # Command 42
CMD_OSCOPE_GET_READINGS = 0x2B                      # Command 43
                                                    
CMD_DMM_GET_READINGS = 0x31                         # Command 49

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# Safe Operating Ranges for H9 HET
# DISCHARGE_VOLTAGE_MIN = 300.0                       # Volts
# DISCHARGE_VOLTAGE_MAX = 400.0                       # Volts

# DISCHARGE_CURRENT_MIN = 15.0                        # Amps
# DISCHARGE_CURRENT_MAX = 30.0                        # Amps

# MASS_FLOW_MIN = 10.0                                # mg/sec
# MASS_FLOW_MIN = 20.0                                # mg/sec

# MAGNET_PERCENT_MIN = 75.0                           # %
# MAGNET_PERCENT_MAX = 125.0                          # %
# ---------------------------------------------------------------------------



# Data Structures 

@dataclass
class MagnaReadings:
    voltage: float
    current: float
    enabled: bool
    voltage_limit: float
    current_limit: float
    overvoltage_trip: float
    overcurrent_trip: float
    local_control: bool
    alarm: bool

@dataclass
class MagnaControl:
    voltage_limit: float
    current_limit: float
    overvoltage_trip: float
    overcurrent_trip: float
    enable: bool

@dataclass
class AlicatReadings:
    label: str
    gas: str
    setpoint: float
    setpoint_units: str
    mass_flow: float
    mass_flow_units: str
    pressure: float
    pressure_units: str
    temperature: float
    temperature_units: str
    volume_flow: float
    volume_flow_units: str
    valve_hold: bool

@dataclass
class AlicatControl:
    label: str
    setpoint: float
    units: str
    loop_control_variable: int=0            # U16 ENUM (unsigned word - 16 bits): 0 = Mass Flow, 1 = | Pressure |, 2 = Volume Flow
    valve_hold: bool = False

@dataclass
class LambdaReadings:
    label: str
    voltage: float
    current: float
    enable: bool
    voltage_limit: float
    current_limit: float
    overvoltage_protection: float
    remote_mode: int                        # U8 ENUM (unsigned byte - unsigned 8 bit int): 0 = Local, 1 = Remote, 2 = Local Lockout
    fault: bool

@dataclass
class LambdaControl:
    label: str
    voltage_limit: float
    current_limit: float
    overvoltage_protection: float
    enable: bool = False

@dataclass
class KeysightDMMReadings:
    current: float

@dataclass
class OscopeAxis:
    increment: float
    origin: float
    reference: int

@dataclass
class OscopeWaveform:
    x: OscopeAxis
    y: OscopeAxis
    data: list[int]

    # Return physical x-axis values for each waveform point.
    # For Keysight-style waveform scaling: 
    #           time[i] = (i - x_reference) * x_increment + x_origin
    def time_values(self) -> list[float]:
        if len(self.data) > 0:
            time = np.array([
                (i - self.x.reference) * self.x.increment + self.x.origin
                for i in range(len(self.data))
            ])
            time -= time[0]
        else:
            time = np.array([])
        return time

    # For Keysight-style waveform scaling:
    #           signal[i] = (raw[i] - y_reference) * y_increment + y_origin
    def y_values(self) -> list[float]:
        return [
            (float(raw_point) - self.y.reference) * self.y.increment + self.y.origin
            for raw_point in self.data
        ]
    
    @property
    def sample_rate(self) -> float | None:
        if self.x.increment == 0:
            return None
        return 1.0 / abs(self.x.increment)
    
    @property
    def duration(self) -> float:
        if len(self.data) <= 1:
            return 0.0
        return(len(self.data) - 1) * abs(self.x.increment)
        
@dataclass
class OscopeReadings:
    label: str
    peak_to_peak: float
    rms: float
    average: float
    waveform: OscopeWaveform

@dataclass
class OscopeConfig:
    label: str
    range: float
    offset: float
    collect_waveforms: True

@dataclass
class DeviceCommands:
    magna_supplies: MagnaControl
    alicat_supplies: list[AlicatControl]
    lambda_supplies: list[LambdaControl]

# Flattened Binary Reader/Writer Functions
class LabViewReader:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.offset = 0

    def remaining(self) -> int:
        payload = self.payload
        offset = self.offset
        return len(payload) - offset
    
    def read_payload(self, n: int) -> bytes:
        payload = self.payload
        offset = self.offset

        if offset + n > len(payload):
            raise ValueError(
                f"Payload ended early. Need {n} bytes at offset {offset}, but only {self.remaining()} bytes remain."
            )
        
        out = payload[offset:offset+n]
        self.offset +=n
        return out

    # Signed 32 bit integer (big-endian)
    def i32(self) -> int:
        return struct.unpack(">i", self.read_payload(4))[0]
    
    # Unsigned 8 bit integer (big-endian)
    def u8(self) -> int:
        return struct.unpack(">B", self.read_payload(1))[0]
    
    # Unsigned 16 bit integer (big-endian)
    def u16(self) -> int:
        return struct.unpack(">H", self.read_payload(2))[0]
    
    # Unsigned 32 bit integer (big_endian)
    def u32(self) -> int:
        return struct.unpack(">I", self.read_payload(4))[0]
    
    # 64-bit float (big-endian)
    def f64(self) -> float:
        return struct.unpack(">d", self.read_payload(8))[0]
    
    # Boolean (big-endian)
    def boolean(self) -> bool:
        return self.u8() != 0
    
    # String (big-endian): 
    # First 4 bytes is length N, followed by N bytes of UTF-8 encoded string
    def string(self) -> str:
        length = self.i32()
        
        if length < 0:
            raise ValueError(
                f"Negative string length: {length}"
            )
        raw_bytes = self.read_payload(length)

        # Throw an error if the bytes aren't valid UTF-8
        # Replace invalid sequences with the Unicode replacement character instead of crashing
        return raw_bytes.decode("utf-8", errors="replace")

    def array_length(self, context:str) -> int:
        length = self.i32()
        if length < 0:
            raise ValueError(f"{context}: negative array length = {length}")
        return length

    def assert_consume_all(self, context: str) -> None:
        if self.remaining() != 0:
            extra = self.payload[self.offset:]
            raise ValueError(
                f"{context}: decoded payload but {self.remaining()} bytes remain unconsumed. "
                f"Number of Extra Bytes: {extra.hex(' ')}"
                )
    
class LabViewWriter:
    def __init__(self):
        self.value_types: list[bytes] = []

    def bytes(self) -> bytes:
        return b"".join(self.value_types)
    
    def i32(self, value: int) -> None:
        self.value_types.append(struct.pack(">i", int(value)))

    def u8(self, value: int) -> None:
        self.value_types.append(struct.pack(">B", int(value)))

    def u16(self, value: int) -> None:
        self.value_types.append(struct.pack(">H", int(value)))

    def u32(self, value: int) -> None:
        self.value_types.append(struct.pack(">I", int(value)))

    def f64(self, value: float) -> None:
        self.value_types.append(struct.pack(">d", float(value)))

    def boolean(self, value: bool) -> None:
        self.u8(1 if value else 0)

    def string(self, value: str) -> None:
        encoded = str(value).encode("utf-8")
        self.i32(len(encoded))
        self.value_types.append(encoded)

def flatten_empty_string() -> bytes:
    writer = LabViewWriter()
    writer.string("")
    return writer.bytes()

def empty_payload() -> bytes:
    return flatten_empty_string() if SEND_EMPTY_ARG else b""

def unpack_dmm_readings(payload: bytes) -> KeysightDMMReadings:
    reader = LabViewReader(payload)
    output = KeysightDMMReadings(
        current = reader.f64()
    )

    reader.assert_consume_all("Keysight DMM readings")
    return output

# PEPL Lab Device Specific Unpacking Functions
def unpack_magna_readings(payload: bytes) -> MagnaReadings:
    reader = LabViewReader(payload)

    output = MagnaReadings(
        voltage = reader.f64(),
        current = reader.f64(),
        enabled = reader.boolean(),
        voltage_limit = reader.f64(),
        current_limit = reader.f64(),
        overvoltage_trip = reader.f64(),
        overcurrent_trip = reader.f64(),
        local_control = reader.boolean(),
        alarm = reader.boolean(),
    )

    reader.assert_consume_all("Magna Readings")
    return output

def unpack_alicat_readings(payload: bytes) -> list[AlicatReadings]:
    reader = LabViewReader(payload)
    
    # Array of Clusters
    n_controllers = reader.array_length("Alicat Readings")

    output: list[AlicatReadings] = []
    for _ in range(n_controllers):
        output.append(
            AlicatReadings(
                label = reader.string(),
                gas = reader.string(),
                setpoint = reader.f64(),
                setpoint_units = reader.string(),
                mass_flow = reader.f64(),
                mass_flow_units = reader.string(),
                pressure = reader.f64(),
                pressure_units = reader.string(),
                temperature = reader.f64(),
                temperature_units = reader.string(),
                volume_flow = reader.f64(),
                volume_flow_units = reader.string(),
                valve_hold = reader.boolean(),
            )
        )

    reader.assert_consume_all("Alicat Readings")
    return output

def unpack_lambda_readings(payload: bytes) -> list[LambdaReadings]:
    reader = LabViewReader(payload)
    
    # Array of Clusters
    n_supplies = reader.array_length("Lambda Readings")

    output: list[LambdaReadings] = []
    for _ in range(n_supplies):
        output.append(
            LambdaReadings(
                label = reader.string(),
                voltage = reader.f64(),
                current = reader.f64(),
                enable = reader.boolean(),
                voltage_limit = reader.f64(),
                current_limit = reader.f64(),
                overvoltage_protection = reader.f64(),
                remote_mode = reader.u8(),
                fault = reader.boolean(),
            )
        )

    reader.assert_consume_all("Lambda Readings")
    return output


def unpack_oscope_waveform(reader: LabViewReader) -> OscopeWaveform:
    # Waveform Cluster from LabVIEW:
    #   X Cluster: X increment (Double), X origin (Double), X Reference (U32)
    #   Y Cluster: Y increment (Double), Y origin (Double), Y Reference (U32)
    #   Data: 1D array (U16 points)
    
    x_axis = OscopeAxis(
        increment = reader.f64(),
        origin = reader.f64(),
        reference = reader.u32(),
    )

    y_axis = OscopeAxis(
        increment = reader.f64(),
        origin = reader.f64(),
        reference = reader.u32(),
    )

    n_points = reader.array_length("Oscope Wavefrom Data")
    bytes = reader.read_payload(n_points)
    dt = np.dtype(np.uint8)
    dt = dt.newbyteorder('>')
    data = np.frombuffer(bytes, dtype=dt)
    return OscopeWaveform(x = x_axis, y = y_axis, data = data)

def unpack_oscope_readings(payload: bytes) -> list[OscopeReadings]:
    reader = LabViewReader(payload)

    # 1-D Array of Keysight O-Scope Single Readings.ctl" Clusters
    n_readings = reader.array_length("Oscope Readings")

    output: list[OscopeReadings] = []

    for _ in range(n_readings):
        output.append(
            OscopeReadings(
                label = reader.string(),
                peak_to_peak = reader.f64(),
                rms = reader.f64(),
                average = reader.f64(),
                waveform = unpack_oscope_waveform(reader),
            )
        )

    reader.assert_consume_all("Oscope Readings")

    return output

# PEPL Lab Device Specific Packing Functions
def pack_magna_control(control: MagnaControl, verbose=False) -> bytes:
    writer = LabViewWriter()

    writer.f64(control.voltage_limit)
    writer.f64(control.current_limit)
    writer.f64(control.overvoltage_trip)
    writer.f64(control.overcurrent_trip)
    writer.boolean(control.enable)

    bytes = writer.bytes()
    if verbose:
        print("Magna bytes:", bytes)
    
    return bytes

def pack_oscope_config(controls: list[OscopeConfig], verbose=False) -> bytes:
    writer = LabViewWriter()
    writer.i32(len(controls))

    for c in controls:
        writer.string(c.label)
        writer.f64(c.range)
        writer.f64(c.offset)
        writer.boolean(c.collect_waveforms)

    bytes = writer.bytes()
    if verbose:
        print("Oscope config bytes:", bytes)
    
    return bytes

def pack_alicat_control(controls: list[AlicatControl], verbose=False) -> bytes:
    writer = LabViewWriter()

    writer.i32(len(controls))

    for c in controls:
        writer.string(c.label)
        writer.f64(c.setpoint)
        writer.string(c.units)
        writer.u16(c.loop_control_variable)
        writer.boolean(c.valve_hold)

    bytes = writer.bytes()
    if verbose:
        print("Alicat bytes:", bytes)
    
    return bytes

def pack_lambda_control(controls: list[LambdaControl], verbose=False) -> bytes:
    writer = LabViewWriter()

    writer.i32(len(controls))
    
    for c in controls:
        writer.string(c.label)
        writer.f64(c.voltage_limit)
        writer.f64(c.current_limit)
        writer.f64(c.overvoltage_protection)
        writer.boolean(c.enable)

    bytes = writer.bytes()
    if verbose:
        print("Lambda bytes:", bytes)
    
    return bytes
    

# TCP Client 
def receive_from_labview(socket: socket.socket, n_bytes: int) -> bytes:
    data = b""

    while len(data) < n_bytes:
            packet = socket.recv(n_bytes - len(data))
            if not packet:
                raise ConnectionError(
                    f"Socket closed early. Expected {n_bytes} bytes. "
                    f"Only received {len(data)} bytes before connection closed."
                )
            
            data += packet

    return data

class LabViewClient:
    
    def __init__(self,
                 host: str = LABVIEW_IP,
                 port: int = LABVIEW_PORT,
                 timeout: float = SOCKET_TIMEOUT
                ):
        
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket: socket.socket | None = None

    def connect(self) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(self.timeout)
        self.socket.connect((self.host, self.port))
        
        print(
            f"Connected to LabVIEW at {self.host}:{self.port}"
        )

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def __enter__(self) -> "LabViewClient":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def send_packet(self, command_id: int, payload: bytes = b"") -> None:
        if self.socket is None:
            raise RuntimeError(
                "Socket is not connected."
            )
        
        header = struct.pack(HEADER_FMT, command_id, len(payload))
        self.socket.sendall(header + payload)

    def receive_packet(self, expected_command_id: int | None = None) -> tuple[int, bytes]:
        if self.socket is None:
            raise RuntimeError(
                "Socket is not connected."
            )
        
        header = receive_from_labview(self.socket, HEADER_LENGTH)
        command_id, payload_length = struct.unpack(HEADER_FMT, header)

        if payload_length < 0:
            raise ValueError(
                f"LabVIEW returned negative payload length: {payload_length}"
            )
        
        payload = receive_from_labview(self.socket, payload_length)

        if expected_command_id is not None and command_id != expected_command_id:
            raise ValueError(
                f"Unexpected command ID. Expected {expected_command_id}, but got {command_id}"
            )
        
        return command_id, payload
    
    def request(self, command_id: int, payload: bytes = b"") -> bytes:
        self.send_packet(command_id, payload)
        _, response_payload = self.receive_packet(expected_command_id = command_id)
        return response_payload
    


# API Handling

def check_empty_ack(name: str, response_payload: bytes) -> None:
    # LabVIEW acknowledges a set command with:
    # Payload length 0
    # Flattened Empty String: 00 00 00 00
    # Accept Both
    if response_payload in (b"", b"\x00\x00\x00\x00"):
        return
    print(
        f"Warning: {name} returned unexpected non-empty payload "
        f"({len(response_payload)}) bytes: {response_payload.hex(' ')}"
    )

def get_magna_readings(client: LabViewClient) -> MagnaReadings:
    payload = client.request(CMD_MAGNA_GET_READINGS, empty_payload())
    return unpack_magna_readings(payload)
    
def set_magna_control(client: LabViewClient, control: MagnaControl, verbose=False) -> None:
    response = client.request(CMD_MAGNA_SET_CONTROL, pack_magna_control(control, verbose))
    return check_empty_ack("Magna Set Control", response)

def get_alicat_readings(client: LabViewClient) -> list[AlicatReadings]:
    payload = client.request(CMD_ALICAT_GET_READINGS, empty_payload())
    return unpack_alicat_readings(payload)

def set_alicat_control(client: LabViewClient, control: list[AlicatControl], verbose=False) -> None:
    response = client.request(CMD_ALICAT_SET_CONTROL, pack_alicat_control(control, verbose))
    return check_empty_ack("Alicat Set Control", response)

def get_lambda_readings(client: LabViewClient) -> list[LambdaReadings]:
    payload = client.request(CMD_LAMBDA_GET_READINGS, empty_payload())
    return unpack_lambda_readings(payload)

def set_lambda_control(client: LabViewClient, control: list[LambdaControl], verbose=False) -> None:
    response = client.request(CMD_LAMBDA_SET_CONTROL, pack_lambda_control(control, verbose))
    return check_empty_ack("Lambda Set Controls", response)

def get_oscope_readings(client: LabViewClient) -> list[OscopeReadings]:
    payload = client.request(CMD_OSCOPE_GET_READINGS, empty_payload())
    return unpack_oscope_readings(payload)

def get_dmm_readings(client: LabViewClient) -> KeysightDMMReadings:
    payload = client.request(CMD_DMM_GET_READINGS, empty_payload())
    return unpack_dmm_readings(payload)

def set_oscope_config(client: LabViewClient, config: list[OscopeConfig]) -> None:
    response = client.request(CMD_OSCOPE_SET_CONFIG, pack_oscope_config(config))
    return check_empty_ack("Oscope set config", response)

# Raw LabVIEW Collection
def get_all_readings(client: LabViewClient, *, include_oscope: bool = True) -> dict[str, Any]:
    magna_supplies = get_magna_readings(client)
    alicat_supplies = get_alicat_readings(client)
    lambda_supplies = get_lambda_readings(client)

    packet: dict[str, Any] = {
        "timestamp": time.time(),
        "magna": asdict(magna_supplies),
        "alicat": [asdict(item) for item in alicat_supplies],
        "lambda": [asdict(item) for item in lambda_supplies],
    }

    if include_oscope:
        oscope = get_oscope_readings(client)
        packet["oscope"] = [asdict(item) for item in oscope]


    return packet


# Label Matching

ANODE_ALICAT_LABEL = {
    "anode",
}

CATHODE_ALICAT_LABLE = {
    "cathode"
}

OUTER_MAGNET_LAMBDA = {
    "outer"
    "outer magnet"
    "outer_magnet"
}

INNER_MAGNET_LAMBDA ={
    "inner"
    "inner magnet"
    "inner_magnet"
}

OSCOPE_DISCHARGE_CURRENT = {
    "discharge current"
    "discharge_current"
}

OSCOPE_PLASMA_POTENTIAL = {
    "plasma potenital"
    "plasma_potential"
}

def normalize_categorization_label(label: str) -> str:
    return "".join(ch.lower() for ch in str(label) if ch.isalnum())

def get_field(item: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(field_name, default)
    return getattr(item, field_name, default)

def find_by_label(items: list[Any], aliases: tuple[str, ...], device_kind: str) -> Any:
    desired = {normalize_categorization_label(alias) for alias in aliases}

    for item in items:
        item_label = str(get_field(item, "label", ""))
        if normalize_categorization_label(item_label) in desired:
            return item
        
        avaiable_labels = [str(get_field(item, "label", "<missing label>"))]
        raise ValueError(
            f"Cound not find {device_kind} mathcing alises {aliases}. "
            f"Avaiable labels from LabVIEW: {avaiable_labels}. "
            f"Update the corresponding **Aliases Labels**."
        )

def print_available_labels(readings: dict[str, Any]) -> None:
    print("Alicat Labels:", [str(get_field(item, "label", "")) for item in readings.get("alicat", [])])
    print("Lambda Labels:", [str(get_field(item, "label", "")) for item in readings.get("lambda", [])])
    print("Oscope Labels:", [str(get_field(item, "label", "")) for item in readings.get("oscope", [])])