import numpy as np

import lib.labview as labview
import lib.controls as controls

controller = controls.ThrusterController("h9_calibration.json", "Kr")
num_avg_points = 50

with labview.LabViewClient(timeout=num_avg_points) as client:
    thrust = controller.take_thrust(client, num_avg_pts=num_avg_points)
    print(f"{thrust=:.3f}")
