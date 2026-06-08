import argparse
import os
from pathlib import Path
import pickle
import time

import lib.labview as labview
from pid_mf_dc import PIDState, pid_mass_flow_step

parser = argparse.ArgumentParser()

parser.add_argument("--target-current", "-i", type=float, required=True, help="Target discharge current [A]")
parser.add_argument("--nominal-flow", "-f", type=float, required=True, help="Nominal anode flow setpoint")
parser.add_argument("--min-flow", type=float, required=True, help="Minimum allowed anode flow setpoint")
parser.add_argument("--max-flow", type=float, required=True, help="Maximum allowed anode flow setpoint")

parser.add_argument("--kp", type=float, required=True, help="Proportional gain")
parser.add_argument("--ki", type=float, default=0.0, help="Integral gain")
parser.add_argument("--kd", type=float, default=0.0, help="Derivative gain")

parser.add_argument("--sample-time", "-t", type=float, default=3.0, help="PID loop period [s]")
parser.add_argument("--run-time", type=float, default=30.0, help="Total PID run time [s]")

parser.add_argument("--host-ip", type=str, default=labview.LABVIEW_IP, help="LabVIEW IP address")
parser.add_argument("--port", type=int, default=labview.LABVIEW_PORT, help="LabVIEW TCP port")
parser.add_argument("--verbose", "-v", action="store_true", help="Print PID loop data")

parser.add_argument("--output", "-o", type=Path, default=Path("."), help="Folder where PID data is saved")
parser.add_argument("--prefix", "-p", type=str, default="pid_mass_flow", help="Output filename prefix")



def find_anode_alicat(alicat_readings):
    for item in alicat_readings:
        label = item.label.lower().replace("_", "").replace(" ", "")
        if "anode" in label:
            return item

    labels = [item.label for item in alicat_readings]
    raise ValueError(f"Could not find anode Alicat. Available Alicat labels: {labels}")



def send_anode_flow(client, anode_flow_command):
    alicat_readings = labview.get_alicat_readings(client)
    anode = find_anode_alicat(alicat_readings)

    controls = []

    for item in alicat_readings:
        if item.label == anode.label:
            setpoint = anode_flow_command
        else:
            setpoint = item.setpoint

        controls.append(
            labview.AlicatControl(
                label=item.label,
                setpoint=setpoint,
                units=item.setpoint_units,
                loop_control_variable=0,            # 0 = mass flow
                valve_hold=item.valve_hold,
            )
        )

    labview.set_alicat_control(client, controls)
    return anode



def main(args):
    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)

    out_file = output_dir / f"{args.prefix}.pkl"

    pid_state = PIDState()
    data_log = []

    print("Starting PID mass-flow control.")
    print(f"Target current: {args.target_current} A")
    print(f"Nominal flow:   {args.nominal_flow}")
    print(f"Flow limits:    [{args.min_flow}, {args.max_flow}]")
    print(f"PID gains:      kp={args.kp}, ki={args.ki}, kd={args.kd}")
    print(f"Run time:       {args.run_time} s")
    print(f"Sample time:    {args.sample_time} s")

    start_time = time.monotonic()



    with labview.LabViewClient(host=args.host_ip, port=args.port) as client:
        while True:
            loop_start = time.monotonic()
            elapsed_time = loop_start - start_time

            if elapsed_time >= args.run_time:
                break

            # Read discharge current from Magna
            magna = labview.get_magna_readings(client)
            measured_current = magna.current

            # Run one PID step
            anode_flow_command = pid_mass_flow_step(
                target_current=args.target_current,
                measured_current=measured_current,
                nominal_flow=args.nominal_flow,
                dt=args.sample_time,
                state=pid_state,
                kp=args.kp,
                ki=args.ki,
                kd=args.kd,
                min_flow=args.min_flow,
                max_flow=args.max_flow,
            )

            # Send new anode flow setpoint
            old_anode = send_anode_flow(client, anode_flow_command)

            # Log
            error = args.target_current - measured_current

            sample = {
                "time_seconds": elapsed_time,
                "target_current_amps": args.target_current,
                "measured_current_amps": measured_current,
                "error_aamps": error,
                "old_anode_setpoint": old_anode.setpoint,
                "old_anode_mass_flow": old_anode.mass_flow,
                "new_anode_setpoint": anode_flow_command,
                "anode_units": old_anode.setpoint_units,
            }

            data_log.append(sample)

            if args.verbose:
                print(
                    f"t={elapsed_time:7.2f} s | "
                    f"target={args.target_current:8.3f} A | "
                    f"measured={measured_current:8.3f} A | "
                    f"error={error:8.3f} A | "
                    f"flow_cmd={anode_flow_command:8.3f} {old_anode.setpoint_units}"
                )

            # Maintain sample time
            loop_time = time.monotonic() - loop_start
            sleep_time = args.sample_time - loop_time

            if sleep_time > 0:
                time.sleep(sleep_time)



    with open(out_file, "wb") as fd:
        pickle.dump(data_log, fd)

    print(f"PID data written to {out_file}")



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)



