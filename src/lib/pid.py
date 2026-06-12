from __future__ import annotations
from dataclasses import dataclass

@dataclass
class PIDState:

    integral_error: float = 0.0
    previous_error: float | None = None

def pid_mass_flow_step(
    *,
    target_current: float,
    measured_current: float,
    nominal_flow: float,
    dt: float,
    state: PIDState,
    kp: float,
    Ti: float,
    Td: float,
    min_flow: float,
    max_flow: float,
    control_direction: float = 1.0,
    integral_limit: float | None = None,
) -> float:

    print(f"{nominal_flow=}, {min_flow=}, {max_flow=}")

    if dt <= 0.0:
        raise ValueError("dt must be greater than zero")

    if min_flow == 0.0:
        raise ValueError("set a minimum flow value")
    
    if max_flow == 0.0:
        raise ValueError("set maximum flow value")
    
    if min_flow >= max_flow:
        raise ValueError("minimum flow must be less than maximum flow")
    
    if control_direction not in (-1.0, 1.0):
        raise ValueError("control_direction should be either +1.0 or -1.0")
    
    error = target_current - measured_current

    if state.previous_error is None:
        derivative_error = 0.0
    else:
        derivative_error = (error - state.previous_error) / dt

    candidate_integral = state.integral_error + error * dt

    if integral_limit is not None:
        candidate_integral = max(
            -integral_limit, min(integral_limit, candidate_integral)
        )

    ki = kp / Ti
    kd = kp * Td

    proportional = kp * error
    integral = ki * candidate_integral
    derivative = kd * derivative_error

    correction = control_direction * (proportional + integral + derivative)
    unclamped_flow = nominal_flow + correction

    # Clamp Final Flow for Safety
    flow = max(
        min_flow, min(max_flow, unclamped_flow)
    )

    # Integral Anti-windup
    saturated_high = unclamped_flow > max_flow
    saturated_low = unclamped_flow < min_flow

    integral_push_up = control_direction * ki * error > 0.0
    integral_push_down = control_direction * ki * error < 0.0

    freeze_integral = (
        (saturated_high and integral_push_up)
        or
        (saturated_low and integral_push_down)
    )

    if not freeze_integral:
        state.integral_error = candidate_integral

    state.previous_error = error

    return flow