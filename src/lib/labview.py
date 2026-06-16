from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dataclasses import dataclass, asdict, fields, is_dataclass
from typing import Any, Literal, Annotated, get_args, get_origin, get_type_hints

import socket, struct

from enum import Enum

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
CMD_MAGNA_GET_READINGS  = 18
CMD_MAGNA_SET_CONTROL   = 19

CMD_ALICAT_FETCH        = 24
CMD_ALICAT_SET_COMMS    = 25
CMD_ALICAT_SET_CONFIG   = 26                  
CMD_ALICAT_GET_READINGS = 27
CMD_ALICAT_SET_CONTROL  = 28

CMD_LAMBDA_GET_READINGS = 35
CMD_LAMBDA_SET_CONTROL  = 36

CMD_OSCOPE_SET_CONFIG   = 42
CMD_OSCOPE_GET_READINGS = 43
                                                    
CMD_DMM_GET_READINGS    = 49

CMD_THRUSTSTAND_FETCH             = 56
CMD_THRUSTSTAND_SET_COMMUNICATION = 57
CMD_THRUSTSTAND_SET_CONFIG        = 58                
CMD_THRUSTSTAND_GET_READINGS      = 59

# Data Structures
class IntWidth(Enum):
    U8 = "u8"
    U16 = "u16"
    U32 = "u32"
    I8 = "i8"
    I16 = "i16"
    I32 = "i32"

class FloatWidth(Enum):
    F32 = "f32"
    F64 = "f64"

U8, U16, U32 = IntWidth.U8, IntWidth.U16, IntWidth.U32,
I8, I16, I32 = IntWidth.I8, IntWidth.I16, IntWidth.I32,
F32, F64 = FloatWidth.F32, FloatWidth.F64

# ---------------------------------------------------------------------------
# Generic pack/unpack based on dataclass field type hints.
#
# Plain `int` defaults to i32 and plain `float` defaults to f64. Use
# Annotated[int, U8/U16/U32/I8/I16/I32] or Annotated[float, F32/F64] to
# override. Nested dataclasses, list[T] (array of clusters), and
# NDArray[dtype] fields (length-prefixed arrays) are handled automatically.

def _unwrap_annotated(tp: Any) -> tuple[Any, tuple[Any, ...]]:
    if get_origin(tp) is Annotated:
        base, *metadata = get_args(tp)
        return base, tuple(metadata)
    return tp, ()

def _ndarray_dtype(tp: Any) -> Any:
    _, dtype_arg = get_args(tp)
    (dtype,) = get_args(dtype_arg)
    return dtype

def pack_struct(value, tp: Any) -> bytes:
    writer = LabViewWriter()
    writer.pack_value(value, tp)
    return writer.bytes()

def unpack_struct(payload: bytes, tp: Any) -> Any:
    reader = LabViewReader(payload)
    result = reader.unpack_value(tp)
    reader.assert_consume_all(str(tp))
    return result

# ---------------------------------------------------------------------------

@dataclass
class MagnaReadings:
    voltage: Annotated[float, F64]
    current: Annotated[float, F64]
    enabled: bool
    voltage_limit: Annotated[float, F64]
    current_limit: Annotated[float, F64]
    overvoltage_trip: Annotated[float, F64]
    overcurrent_trip: Annotated[float, F64]
    local_control: bool
    alarm: bool

@dataclass
class MagnaControl:
    voltage_limit: Annotated[float, F64]
    current_limit: Annotated[float, F64]
    overvoltage_trip: Annotated[float, F64]
    overcurrent_trip: Annotated[float, F64]
    enable: bool

GasType = Literal["Air", "Ar", "CO2", "H2", "He", "N2", "N2O", "Ne", "O2", "Kr", "Xe"] 
GAS_INDICES = {g: i for (i, g) in enumerate(get_args(GasType))}

@dataclass
class AlicatDeviceComms:
    label: str
    id: Annotated[int, U8]

@dataclass
class AlicatCommunications:
    hub_address: str
    port: Annotated[int, U16]
    connection: Annotated[int, I32]
    devices: list[AlicatDeviceComms]

@dataclass
class AlicatConfig:
    label: str
    gas:  Annotated[int, U16]
    remote_lockout: bool = False

@dataclass
class AlicatControl:
    label: str
    setpoint: Annotated[float, F64]
    valve_hold: bool = False

# class AlicatReadings:
#     label: str
#     gas: str
#     setpoint: float
#     setpoint_units: str
#     mass_flow: float
#     mass_flow_units: str
#     pressure: float
#     pressure_units: str
#     temperature: float
#     temperature_units: str
#     volume_flow: float
#     volume_flow_units: str
#     valve_hold: bool

@dataclass
class AlicatReadings:
    label: str
    gas: str
    setpoint: Annotated[float, F64]
    mass_flow: Annotated[float, F64]
    pressure: Annotated[float, F64]
    temperature: Annotated[float, F64]
    volume_flow: Annotated[float, F64]
    valve_hold: bool

@dataclass
class LambdaReadings:
    label: str
    voltage: Annotated[float, F64]
    current: Annotated[float, F64]
    enable: bool
    voltage_limit: Annotated[float, F64]
    current_limit: Annotated[float, F64]
    overvoltage_protection: Annotated[float, F64]
    # U8 ENUM: 0 = Local, 1 = Remote, 2 = Local Lockout
    remote_mode: Annotated[int, U8]         
    fault: bool

@dataclass
class LambdaControl:
    label: str
    voltage_limit: Annotated[float, F64]
    current_limit: Annotated[float, F64]
    overvoltage_protection: Annotated[float, F64]
    enable: bool = False

@dataclass
class KeysightDMMReadings:
    current: float

@dataclass
class OscopeAxis:
    increment: Annotated[float, F64]
    origin: Annotated[float, F64]
    reference: Annotated[int, U32]
    
    @staticmethod
    def from_dict(d):
        return OscopeAxis(
            increment = d["increment"],
            origin = d["origin"],
            reference = d["reference"],
        )

@dataclass
class OscopeWaveform:
    x: OscopeAxis
    y: OscopeAxis
    data: NDArray[np.uint8]

    @staticmethod
    def from_dict(d):
        return OscopeWaveform(
            x = OscopeAxis.from_dict(d["x"]),
            y = OscopeAxis.from_dict(d["y"]),
            data = d["data"],
        )

    # Return physical x-axis values for each waveform point.
    # For Keysight-style waveform scaling: 
    #           time[i] = (i - x_reference) * x_increment + x_origin
    def time_values(self) -> np.ndarray:
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
    def y_values(self) -> np.ndarray:
        return np.array([
            (float(raw_point) - self.y.reference) * self.y.increment + self.y.origin
            for raw_point in self.data
        ])
    
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
class OscopeTimeBase:
    range: Annotated[float, F64]
    # 0 for left, 1 for center, 2 for right
    reference: Annotated[int, U8] = 1 
    position: Annotated[float, F64] = 0.0

@dataclass
class OscopeChannelConfig:
    label: str
    range: Annotated[float, F64]
    offset: Annotated[float, F64]
    collect_waveforms: bool

@dataclass
class OscopeConfig:
    time_base: OscopeTimeBase
    channels: list[OscopeChannelConfig]
        
@dataclass
class OscopeReadings:
    label: str
    peak_to_peak: Annotated[float, F64]
    rms: Annotated[float, F64]
    average: Annotated[float, F64]
    waveform: OscopeWaveform

@dataclass
class PIDGain:
    Kp: Annotated[float, F64] 
    Ki: Annotated[float, F64]
    Kd: Annotated[float, F64]

@dataclass
class ThrustStandConfig:
    num_points: Annotated[int, U16]
    gains: PIDGain

@dataclass
class ThrustStandReadings:
    setpoint: NDArray[np.float64]
    input: NDArray[np.float64]
    command: NDArray[np.uint16]
    shunt: NDArray[np.int16]
    tilt: NDArray[np.float64]

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

    # Signed 8 bit integer (big-endian)
    def i8(self) -> int:
        return struct.unpack(">b", self.read_payload(1))[0]

    # Signed 16 bit integer (big-endian)
    def i16(self) -> int:
        return struct.unpack(">h", self.read_payload(2))[0]

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

    # 32-bit float (big-endian)
    def f32(self) -> float:
        return struct.unpack(">f", self.read_payload(4))[0]

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

    def read_array(self, dtype: np.typing.DTypeLike, context: str) -> np.ndarray:
        length = self.array_length(context)
        dt = np.dtype(dtype)
        dt = dt.newbyteorder('>')
        bytes = self.read_payload(length * dt.itemsize)
        data = np.frombuffer(bytes, dtype=dt)
        return data

    def assert_consume_all(self, context: str) -> None:
        if self.remaining() != 0:
            extra = self.payload[self.offset:]
            raise ValueError(
                f"{context}: decoded payload but {self.remaining()} bytes remain unconsumed. "
                f"Number of Extra Bytes: {extra.hex(' ')}"
                )

    def unpack_dataclass(self, cls: type) -> Any:
        hints = get_type_hints(cls, include_extras=True)
        kwargs = {
            field.name: self.unpack_value(hints[field.name])
            for field in fields(cls)
        }
        return cls(**kwargs)

    def unpack_value(self, tp: Any) -> Any:
        base, metadata = _unwrap_annotated(tp)

        for marker in metadata:
            if isinstance(marker, (IntWidth, FloatWidth)):
                return getattr(self, marker.value)()

        if base is float:
            return self.f64()
        elif base is bool:
            return self.boolean()
        elif base is int:
            return self.i32()
        elif base is str:
            return self.string()
        elif is_dataclass(base):
            return self.unpack_dataclass(base) # type:ignore
        elif get_origin(base) is list:
            (item_type,) = get_args(base)
            length = self.array_length("array")
            return [self.unpack_value(item_type) for _ in range(length)]
        elif get_origin(base) is np.ndarray:
            return self.read_array(_ndarray_dtype(base), "array")
        else:
            raise TypeError(f"Don't know how to unpack field of type {tp!r}")
    
class LabViewWriter:
    def __init__(self):
        self.value_types: list[bytes] = []

    def bytes(self) -> bytes:
        return b"".join(self.value_types)
    
    def i8(self, value: int) -> None:
        self.value_types.append(struct.pack(">b", int(value)))

    def i16(self, value: int) -> None:
        self.value_types.append(struct.pack(">h", int(value)))

    def i32(self, value: int) -> None:
        self.value_types.append(struct.pack(">i", int(value)))

    def u8(self, value: int) -> None:
        self.value_types.append(struct.pack(">B", int(value)))

    def u16(self, value: int) -> None:
        self.value_types.append(struct.pack(">H", int(value)))

    def u32(self, value: int) -> None:
        self.value_types.append(struct.pack(">I", int(value)))

    def f32(self, value: float) -> None:
        self.value_types.append(struct.pack(">f", float(value)))

    def f64(self, value: float) -> None:
        self.value_types.append(struct.pack(">d", float(value)))

    def boolean(self, value: bool) -> None:
        self.u8(1 if value else 0)

    def string(self, value: str) -> None:
        encoded = str(value).encode("utf-8")
        self.i32(len(encoded))
        self.value_types.append(encoded)

    # Length-prefixed 1D array, big-endian element encoding
    def array(self, value: np.ndarray, dtype: np.typing.DTypeLike) -> None:
        dt = np.dtype(dtype).newbyteorder('>')
        arr = np.asarray(value, dtype=dt)
        self.i32(len(arr))
        self.value_types.append(arr.tobytes())

    def pack_value(self, value: Any, tp: Any) -> None:
        base, metadata = _unwrap_annotated(tp)

        for marker in metadata:
            if isinstance(marker, (IntWidth, FloatWidth)):
                getattr(self, marker.value)(value)
                return

        if base is float:
            self.f64(value)
        elif base is bool:
            self.boolean(value)
        elif base is int:
            self.i32(value)
        elif base is str:
            self.string(value)
        elif is_dataclass(base):
            self.pack_dataclass(value)
        elif get_origin(base) is list:
            (item_type,) = get_args(base)
            self.i32(len(value))
            for item in value:
                self.pack_value(item, item_type)
        elif get_origin(base) is np.ndarray:
            self.array(value, _ndarray_dtype(base))
        else:
            raise TypeError(f"Don't know how to pack field of type {tp!r}")

    def pack_dataclass(self, obj: Any) -> None:
        hints = get_type_hints(type(obj), include_extras=True)
        for field in fields(obj):
            self.pack_value(getattr(obj, field.name), hints[field.name])

def flatten_empty_string() -> bytes:
    writer = LabViewWriter()
    writer.string("")
    return writer.bytes()

def empty_payload() -> bytes:
    return flatten_empty_string() if SEND_EMPTY_ARG else b""

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
                 timeout: float = SOCKET_TIMEOUT,
                 dummy: bool = False,
                ):
        
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket: socket.socket | None = None
        self.dummy = dummy

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
        if not self.dummy:
            self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if not self.dummy:
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
        if not self.dummy:
            self.send_packet(command_id, payload)
            _, response_payload = self.receive_packet(expected_command_id = command_id)
            return response_payload
        else:
            return b""
    
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
    return unpack_struct(payload, MagnaReadings)
    
def set_magna_control(client: LabViewClient, control: MagnaControl) -> None:
    payload = pack_struct(control, MagnaReadings)
    response = client.request(CMD_MAGNA_SET_CONTROL, payload)
    return check_empty_ack("Magna Set Control", response)

#-------------------------------------------
# Alicat mass flow controllers
#-------------------------------------------

def set_alicat_config(client: LabViewClient, config: list[AlicatConfig]) -> None:
    payload = pack_struct(config, list[AlicatCommunications])
    response = client.request(CMD_ALICAT_SET_CONFIG, payload)
    return check_empty_ack("Alicat Set Config", response)

def set_alicat_control(client: LabViewClient, control: list[AlicatControl]) -> None:
    payload = pack_struct(control, list[AlicatControl])
    response = client.request(CMD_ALICAT_SET_CONTROL, payload)
    return check_empty_ack("Alicat Set Control", response)

def set_alicat_comms(client: LabViewClient, comms: AlicatCommunications):
    payload = pack_struct(comms, AlicatCommunications)
    response = client.request(CMD_ALICAT_SET_COMMS, payload)
    return check_empty_ack("Alicat Set Comms", response)

def get_alicat_comms(client: LabViewClient) -> AlicatCommunications:
    # Send header
    # TODO: fetch can be made more general!
    # Depending on the value we send, we could get different values
    header = LabViewWriter()
    header.u8(0)
    payload = client.request(CMD_ALICAT_FETCH, header.bytes())
    return unpack_struct(payload, AlicatCommunications)

def get_alicat_readings(client: LabViewClient) -> list[AlicatReadings]:
    payload = client.request(CMD_ALICAT_GET_READINGS, empty_payload())
    return unpack_struct(payload, list[AlicatReadings])

#-------------------------------------------
# Lambda power supplies
#-------------------------------------------
def get_lambda_readings(client: LabViewClient) -> list[LambdaReadings]:
    payload = client.request(CMD_LAMBDA_GET_READINGS, empty_payload())
    return unpack_struct(payload, list[LambdaReadings])

def set_lambda_control(client: LabViewClient, control: list[LambdaControl]) -> None:
    payload = pack_struct(control, list[LambdaControl])
    response = client.request(CMD_LAMBDA_SET_CONTROL, payload)
    return check_empty_ack("Lambda Set Controls", response)

#-------------------------------------------
# Keysight oscilloscope
#-------------------------------------------

def get_oscope_readings(client: LabViewClient) -> list[OscopeReadings]:
    payload = client.request(CMD_OSCOPE_GET_READINGS, empty_payload())
    return unpack_struct(payload, list[OscopeReadings])

def set_oscope_config(client: LabViewClient, config: OscopeConfig) -> None:
    payload = pack_struct(config, OscopeConfig)
    response = client.request(CMD_OSCOPE_SET_CONFIG, payload)
    return check_empty_ack("Oscope set config", response)

#-------------------------------------------
# Keysight DMM
#-------------------------------------------

def get_dmm_readings(client: LabViewClient) -> KeysightDMMReadings:
    payload = client.request(CMD_DMM_GET_READINGS, empty_payload())
    return unpack_struct(payload, KeysightDMMReadings)

#-------------------------------------------
# Thrust stand
#-------------------------------------------

def set_thruststand_config(client: LabViewClient, config: ThrustStandConfig) -> None:
    payload = pack_struct(config, ThrustStandConfig)
    response = client.request(CMD_THRUSTSTAND_SET_CONFIG, payload)
    return check_empty_ack("Thrust stand set config", response)

def get_thruststand_readings(client: LabViewClient) -> ThrustStandReadings:
    payload = client.request(CMD_THRUSTSTAND_GET_READINGS, empty_payload())
    return unpack_struct(payload, ThrustStandReadings)