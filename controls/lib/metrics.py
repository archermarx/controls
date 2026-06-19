import warnings
import numpy as np


def _get_field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _normalize_label(label):
    return "".join(ch.lower() for ch in str(label) if ch.isalnum())


def get_anode_current_oscope(data):
    """
    Return the Anode Current oscope reading from the LabVIEW data dictionary.
    """

    if "oscope" not in data:
        raise ValueError("No oscope data found.")

    oscope_data = data["oscope"]

    if "Anode Current" in oscope_data:
        return oscope_data["Anode Current"]

    desired = _normalize_label("Anode Current")

    for key, value in oscope_data.items():
        if _normalize_label(key) == desired:
            return value

    raise ValueError(
        f"Could not find 'Anode Current'. "
        f"Available oscope channels: {list(oscope_data.keys())}"
    )


def get_dmm_current(data):
    """
    Return DMM average discharge current if present.
    """

    if "dmm" not in data:
        return None

    return float(_get_field(data["dmm"], "current"))


def get_oscope_average_current(data):
    """
    Return oscope-reported average anode current.
    """

    oscope = get_anode_current_oscope(data)
    return float(_get_field(oscope, "average"))


def get_anode_current_waveform(data):
    """
    Return the physical Anode Current waveform values from the oscope.
    """

    oscope = get_anode_current_oscope(data)
    waveform = _get_field(oscope, "waveform")

    if waveform is None:
        raise ValueError("Anode Current oscope reading has no waveform.")

    if hasattr(waveform, "y_values"):
        current = np.asarray(waveform.y_values(), dtype=float)

    elif isinstance(waveform, dict):
        raw_data = np.asarray(waveform["data"], dtype=float)
        y_axis = waveform["y"]

        y_increment = float(y_axis["increment"])
        y_origin = float(y_axis["origin"])
        y_reference = float(y_axis["reference"])

        current = (raw_data - y_reference) * y_increment + y_origin

    else:
        raise ValueError("Unsupported waveform format.")

    if current.size == 0:
        raise ValueError("No Anode Current waveform data collected.")

    return current


def check_current_alignment(
    data,
    *,
    max_abs_offset_A=2.0,
    max_rel_offset=0.25,
    require_dmm=True,
    raise_on_failure=True,
):
    """
    Check that DMM average current and oscope average current are reasonably aligned.

    max_abs_offset_A:
        Absolute allowed DMM/oscope average-current difference.

    max_rel_offset:
        Fractional allowed difference relative to DMM current.
        Example: 0.25 means 25%.

    If either the absolute difference OR relative difference is acceptable,
    the sample passes.
    """

    dmm_current = get_dmm_current(data)

    if dmm_current is None:
        message = "No DMM current found, so oscope/DMM alignment cannot be checked."

        if require_dmm:
            raise ValueError(message)

        warnings.warn(message)
        return {
            "has_dmm": False,
            "dmm_current": None,
            "oscope_average": get_oscope_average_current(data),
            "abs_offset_A": None,
            "rel_offset": None,
            "passes": True,
        }

    oscope_average = get_oscope_average_current(data)

    abs_offset = abs(dmm_current - oscope_average)
    rel_offset = abs_offset / max(abs(dmm_current), 1e-12)

    passes = (
        abs_offset <= max_abs_offset_A
        or rel_offset <= max_rel_offset
    )

    info = {
        "has_dmm": True,
        "dmm_current": dmm_current,
        "oscope_average": oscope_average,
        "abs_offset_A": abs_offset,
        "rel_offset": rel_offset,
        "passes": passes,
    }

    if not passes:
        message = (
            "DMM and oscope average current do not align. "
            f"DMM = {dmm_current:.6g} A, "
            f"oscope average = {oscope_average:.6g} A, "
            f"offset = {abs_offset:.6g} A, "
            f"relative offset = {100.0 * rel_offset:.2f}%."
        )

        if raise_on_failure:
            raise ValueError(message)

        warnings.warn(message)

    return info


def get_aligned_anode_current_waveform(
    data,
    *,
    max_abs_offset_A=2.0,
    max_rel_offset=0.25,
    require_dmm=True,
    raise_on_failure=True,
):
    """
    Return oscope waveform shifted so its DC average aligns with the DMM current.

    The oscope waveform is still used for oscillation shape/amplitude, but its
    mean is shifted to match the DMM average current.

    This fixes DC-offset mismatch between the two instruments.
    """

    alignment = check_current_alignment(
        data,
        max_abs_offset_A=max_abs_offset_A,
        max_rel_offset=max_rel_offset,
        require_dmm=require_dmm,
        raise_on_failure=raise_on_failure,
    )

    current = get_anode_current_waveform(data)

    if alignment["has_dmm"]:
        dmm_current = alignment["dmm_current"]
        current_mean = float(np.mean(current))

        current = current - current_mean + dmm_current

    return current, alignment


def get_aligned_average_current(
    data,
    *,
    max_abs_offset_A=2.0,
    max_rel_offset=0.25,
    require_dmm=True,
    raise_on_failure=True,
):
    """
    Return the average current used for normalization.

    Prefer DMM current after checking alignment. If DMM is unavailable and
    require_dmm=False, fall back to oscope average.
    """

    alignment = check_current_alignment(
        data,
        max_abs_offset_A=max_abs_offset_A,
        max_rel_offset=max_rel_offset,
        require_dmm=require_dmm,
        raise_on_failure=raise_on_failure,
    )

    if alignment["has_dmm"]:
        return abs(float(alignment["dmm_current"]))

    return abs(float(alignment["oscope_average"]))


def rms_oscillation(
    data,
    setpoint=None,
    control_vector=None,
    *,
    max_abs_offset_A=2.0,
    max_rel_offset=0.25,
    require_dmm=True,
    raise_on_failure=True,
):
    """
    z = RMS oscillation amplitude of aligned anode current.

    Units: amps.
    Lower is better.
    """

    current, alignment = get_aligned_anode_current_waveform(
        data,
        max_abs_offset_A=max_abs_offset_A,
        max_rel_offset=max_rel_offset,
        require_dmm=require_dmm,
        raise_on_failure=raise_on_failure,
    )

    mean_current = np.mean(current)

    return float(
        np.sqrt(
            np.mean((current - mean_current) ** 2)
        )
    )


def rms_oscillation_percent(
    data,
    setpoint=None,
    control_vector=None,
    *,
    max_abs_offset_A=2.0,
    max_rel_offset=0.25,
    require_dmm=True,
    raise_on_failure=True,
):
    """
    z = RMS oscillation amplitude normalized by aligned average current.

    Units: percent.
    Lower is better.
    """

    rms = rms_oscillation(
        data,
        setpoint=setpoint,
        control_vector=control_vector,
        max_abs_offset_A=max_abs_offset_A,
        max_rel_offset=max_rel_offset,
        require_dmm=require_dmm,
        raise_on_failure=raise_on_failure,
    )

    avg_current = get_aligned_average_current(
        data,
        max_abs_offset_A=max_abs_offset_A,
        max_rel_offset=max_rel_offset,
        require_dmm=require_dmm,
        raise_on_failure=raise_on_failure,
    )

    if avg_current < 1e-12:
        raise ValueError("Aligned average discharge current is too close to zero.")

    return float(100.0 * rms / avg_current)


def peak_to_peak_percent(
    data,
    setpoint=None,
    control_vector=None,
    *,
    max_abs_offset_A=2.0,
    max_rel_offset=0.25,
    require_dmm=True,
    raise_on_failure=True,
):
    """
    z = peak-to-peak anode current oscillation normalized by aligned average current.

    Units: percent.
    Lower is better.
    """

    # Check alignment before using this sample.
    check_current_alignment(
        data,
        max_abs_offset_A=max_abs_offset_A,
        max_rel_offset=max_rel_offset,
        require_dmm=require_dmm,
        raise_on_failure=raise_on_failure,
    )

    oscope = get_anode_current_oscope(data)

    avg_current = get_aligned_average_current(
        data,
        max_abs_offset_A=max_abs_offset_A,
        max_rel_offset=max_rel_offset,
        require_dmm=require_dmm,
        raise_on_failure=raise_on_failure,
    )

    if avg_current < 1e-12:
        raise ValueError("Aligned average discharge current is too close to zero.")

    return float(
        100.0 * float(_get_field(oscope, "peak_to_peak")) / avg_current
    )


def average_current_error(
    data,
    setpoint=None,
    control_vector=None,
    *,
    target_current=None,
    max_abs_offset_A=2.0,
    max_rel_offset=0.25,
    require_dmm=True,
    raise_on_failure=True,
):
    """
    z = absolute error between aligned measured average current and target current.

    Units: amps.
    Lower is better.
    """

    if target_current is None:
        raise ValueError("target_current is required for average_current_error.")

    avg_current = get_aligned_average_current(
        data,
        max_abs_offset_A=max_abs_offset_A,
        max_rel_offset=max_rel_offset,
        require_dmm=require_dmm,
        raise_on_failure=raise_on_failure,
    )

    return float(abs(avg_current - target_current))


def rms_percent_plus_current_error(
    data,
    setpoint=None,
    control_vector=None,
    *,
    target_current=None,
    current_error_weight=10.0,
    max_abs_offset_A=2.0,
    max_rel_offset=0.25,
    require_dmm=True,
    raise_on_failure=True,
):
    """
    z = RMS oscillation percent + weighted average-current error.

    Lower is better.
    """

    if target_current is None:
        raise ValueError("target_current is required for rms_percent_plus_current_error.")

    rms_percent = rms_oscillation_percent(
        data,
        setpoint=setpoint,
        control_vector=control_vector,
        max_abs_offset_A=max_abs_offset_A,
        max_rel_offset=max_rel_offset,
        require_dmm=require_dmm,
        raise_on_failure=raise_on_failure,
    )

    current_error = average_current_error(
        data,
        setpoint=setpoint,
        control_vector=control_vector,
        target_current=target_current,
        max_abs_offset_A=max_abs_offset_A,
        max_rel_offset=max_rel_offset,
        require_dmm=require_dmm,
        raise_on_failure=raise_on_failure,
    )

    return float(
        rms_percent + current_error_weight * current_error
    )


def make_metric(
    metric_name,
    *,
    target_current=None,
    max_abs_offset_A=2.0,
    max_rel_offset=0.25,
    require_dmm=True,
    raise_on_failure=True,
):
    """
    Return one metric function with a consistent call signature:

        metric(data, setpoint=None, control_vector=None) -> z

    This keeps execute_surrogate.py generic.
    """

    common_kwargs = {
        "max_abs_offset_A": max_abs_offset_A,
        "max_rel_offset": max_rel_offset,
        "require_dmm": require_dmm,
        "raise_on_failure": raise_on_failure,
    }

    if metric_name == "rms":
        def metric(data, setpoint=None, control_vector=None):
            return rms_oscillation(
                data,
                setpoint=setpoint,
                control_vector=control_vector,
                **common_kwargs,
            )

        return metric

    if metric_name == "rms_percent":
        def metric(data, setpoint=None, control_vector=None):
            return rms_oscillation_percent(
                data,
                setpoint=setpoint,
                control_vector=control_vector,
                **common_kwargs,
            )

        return metric

    if metric_name == "p2p_percent":
        def metric(data, setpoint=None, control_vector=None):
            return peak_to_peak_percent(
                data,
                setpoint=setpoint,
                control_vector=control_vector,
                **common_kwargs,
            )

        return metric

    if metric_name == "current_error":
        def metric(data, setpoint=None, control_vector=None):
            return average_current_error(
                data,
                setpoint=setpoint,
                control_vector=control_vector,
                target_current=target_current,
                **common_kwargs,
            )

        return metric

    if metric_name == "rms_percent_plus_current_error":
        def metric(data, setpoint=None, control_vector=None):
            return rms_percent_plus_current_error(
                data,
                setpoint=setpoint,
                control_vector=control_vector,
                target_current=target_current,
                **common_kwargs,
            )

        return metric

    raise ValueError(f"Unknown metric name: {metric_name}")