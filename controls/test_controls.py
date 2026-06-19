from lib.labview import LabViewClient
import lib.labview as labview

from lib.controls import ThrusterController, ControlPoint

import time

time_s = 0.0

controller = ThrusterController("h9_calibration.json", "Kr")

with LabViewClient(timeout=30) as client:
    comms = labview.get_alicat_comms(client)
    print(f"{comms=}")

    # Set comms
    comms = labview.AlicatCommunications(
        hub_address=comms.hub_address,
        port=comms.port,
        connection=comms.connection,
        devices = [
            labview.AlicatDeviceComms(
                label="Anode", id=1
            ),
            labview.AlicatDeviceComms(
                label="Cathode", id=5
            )
        ]
    )
    labview.set_alicat_comms(client, comms)

    config_anode = labview.AlicatConfig(
        label="Anode",
        gas = labview.GAS_INDICES["Kr"],
        remote_lockout=False
    )

    config_cathode = labview.AlicatConfig(
        label="Cathode",
        gas = labview.GAS_INDICES["Xe"],
        remote_lockout=True
    )

    time.sleep(time_s)

    labview.set_alicat_config(client, [config_anode, config_cathode])

    time.sleep(time_s)

    print("After: \n")
    comms = labview.get_alicat_comms(client)
    print(f"{comms=}")

    controls = [
        labview.AlicatControl(
            "Anode", 0.0, False
        ),
        labview.AlicatControl(
            "Cathode", 0.0, False
        ),
    ]

    time.sleep(time_s)

    labview.set_alicat_control(client,controls)

    time.sleep(time_s)

    readings = labview.get_alicat_readings(client)

    for r in readings:
        print(f"{r=}")
    
    print()

    control_point = ControlPoint(
        anode_mass_flow_rate_kg_s=0.0,
        cathode_flow_fraction=0.07,
        discharge_voltage_v=0.0,
        magnetic_field_scale=0.0
    )

    controller.control_to(control_point, client, set_alicats=False)

    controller.oscope_time_width = 2e-3
    num_thrust_pts = 20
    data = controller.take_data(
        client, num_thrust_points=num_thrust_pts,
        sources=["oscope", "dmm", "lambda", "magna", "thruststand"])

    for channel_name, channel in data["oscope"].items():
        t_vals = channel.waveform.time_values()
        print(f"{channel_name}: Time delta: {t_vals[-1] - t_vals[0]:.3g} (expected {controller.oscope_time_width:.3g})")

    dmm = data["dmm"]
    print(f"\nDMM current: {dmm['current']}")

    lambda_readings = data["lambda"]

    thrust = data["thrust"]
    print(f"{thrust=}")


