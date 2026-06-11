import lib.labview as labview

with labview.LabViewClient() as client:
    config = labview.ThrustStandConfig(
        num_points = 10,
        gains = labview.PIDGain(
            Kp = 0.01,
            Ki = 0.1,
            Kd = 0.0,
        )
    )

    labview.set_thruststand_config(client, config)

    while response.casefold() != "y":
        response = input("Continue to readings? (y/n): ")
    
    readings = labview.get_thruststand_readings(client)
    for attr in dir(labview.ThrustStandReadings):
        print(f"{attr}: {getattr(readings, attr)}")