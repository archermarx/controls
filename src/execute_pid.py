import argparse
import os
from pathlib import Path
import pickle
import time

import matplotlib.pyplot as plt

import lib.controls as controls
import lib.labview as labview
from lib.pid import PIDState, pid_mass_flow_step

parser = argparse.ArgumentParser()

parser.add_argument("--target-current", "-i", type=float, required=True, help="Target discharge current [A]")
parser.add_argument("--cal-file", "-c", type=Path, help="The path to the thruster calibration file")
parser.add_argument("--nominal-flow", "-f", type=float, required=True, help="Nominal anode flow setpoint")
parser.add_argument("--min-flow", type=float, required=True, help="Minimum allowed anode flow setpoint")
parser.add_argument("--max-flow", type=float, required=True, help="Maximum allowed anode flow setpoint")

parser.add_argument("--kp", type=float, required=True, help="Proportional gain")
parser.add_argument("--Ti", type=float, default=float('inf'), help="Integral time constant")
parser.add_argument("--Td", type=float, default=0.0, help="Derivative gain")

parser.add_argument("--sample-time", "-t", type=float, default=10.0, help="PID loop period [s]")
parser.add_argument("--run-time", type=float, default=120.0, help="Total PID run time [s]")

parser.add_argument("--host-ip", type=str, default=labview.LABVIEW_IP, help="LabVIEW IP address")
parser.add_argument("--port", type=int, default=labview.LABVIEW_PORT, help="LabVIEW TCP port")
parser.add_argument("--verbose", "-v", action="store_true", help="Print PID loop data")

parser.add_argument("--output", "-o", type=Path, default=Path("."), help="Folder where PID data is saved")
parser.add_argument("--prefix", "-p", type=str, default="pid_mass_flow", help="Output filename prefix")

parser.add_argument("--setpoint", "-s", type=str, required=True, help="Setpoint file, used to read normalization info")

def main(args):
    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)

    out_file = output_dir / f"{args.prefix}.pkl"
    i = 0
    while os.path.exists(out_file):
        out_file = output_dir / f"{args.prefix}_{i:02d}.pkl"
        i+=1

    pid_state = PIDState()
    data_log = []

    controller = controls.ThrusterController(args.cal_file, "Kr")

    with open(args.setpoint, "rb") as fd:
        setpoint = controls.ControlPoint.model_validate_json(fd.read())
    setpoint.anode_mass_flow_rate_kg_s = args.nominal_flow / 1e6 # convert mg/s to kg/s

    print("Starting PID mass-flow control.")
    print(f"Target current: {args.target_current} A")
    print(f"Nominal flow:   {args.nominal_flow}")
    print(f"Flow limits:    [{args.min_flow}, {args.max_flow}]")
    print(f"PID gains:      kp={args.kp}, Ti={args.Ti}, Td={args.Td}")
    print(f"Run time:       {args.run_time} s")
    print(f"Sample time:    {args.sample_time} s")

    current_measurement_interval_s = 0.5
    settle_time_s = 20

    start_time = time.monotonic()
    loop_start_time = start_time

    def measure_current(client):
        return labview.get_dmm_readings(client).current

    flow_times = []
    current_times = []
    flow_rates = []
    currents_fine = []

    def plot_pid_control(flow_times, current_times, flow_rates, currents_fine):
        fig, axs = plt.subplots(2, 1, layout="constrained")
        axs[0].axhline(args.target_current, linestyle = '--', color='red', zorder=0)
        axs[0].plot(current_times, currents_fine, zorder=1)
        axs[0].set(title = "Currents measured")

        axs[1].step(flow_times, flow_rates)
        axs[1].set(title = "Flow rates set (mg/s)")
        fig.savefig("pid.png")
        plt.close(fig)

    def measure_current_for_interval(total_time, interval, client, offset=0.0):
        """Measure the DMM current at intervals until a specified total time is reached"""
        t = time.monotonic()
        t_start = t
        while (t - t_start) <= total_time:
            loop_start_time = time.monotonic()
            current = measure_current(client)
            t = time.monotonic()
            currents_fine.append(current)
            current_times.append(t - offset)
            plot_pid_control(flow_times, current_times, flow_rates, currents_fine)
            time.sleep(max(interval - (t - loop_start_time), 0.0))

    with labview.LabViewClient(host=args.host_ip, port=args.port) as client:
        print("Setting initial point and sleeping for 20 seconds")
        controller.control_to(setpoint, client, set_lambdas=False, set_magna=False)
        settle_start_time = time.monotonic()
        flow_times.append(settle_start_time - start_time)
        flow_rates.append(setpoint.anode_mass_flow_rate_kg_s * 1e6)
        measure_current_for_interval(
            settle_time_s,
            interval=current_measurement_interval_s,
            client=client,
            offset=settle_start_time
        )

        while True:
            mdot_a = setpoint.anode_mass_flow_rate_kg_s
            print(f"Commanding to {mdot_a}")

            assert mdot_a >= args.min_flow / 1e6
            assert mdot_a <= args.max_flow / 1e6

            loop_start_time = time.monotonic()
            controller.control_to(setpoint, client)
            flow_rates.append(mdot_a * 1e6)
            flow_times.append(loop_start_time - settle_start_time)

            elapsed_time = loop_start_time - start_time
            if elapsed_time >= args.run_time:
                break

            measure_current_for_interval(
                args.sample_time,
                interval=current_measurement_interval_s,
                client=client,
                offset=settle_start_time
            )
            measured_current = currents_fine[-1]
            dt = time.monotonic() - loop_start_time

            print(f"Measured current = {measured_current}")

            # Run one PID step
            anode_flow_command = pid_mass_flow_step(
                target_current=args.target_current,
                measured_current=measured_current,
                nominal_flow=mdot_a*1e6,
                dt=dt, #args.sample_time,
                state=pid_state,
                kp=args.kp,
                Ti=args.Ti,
                Td=args.Td,
                min_flow=args.min_flow,
                max_flow=args.max_flow,
            ) / 1e6

            # Send new anode flow setpoint
            old_anode_flow = mdot_a
            setpoint.anode_mass_flow_rate_kg_s = anode_flow_command

            # Log
            error = args.target_current - measured_current

            sample = {
                "time_seconds": elapsed_time,
                "target_current_amps": args.target_current,
                "measured_current_amps": measured_current,
                "error_amps": error,
                "old_anode_mass_flow": old_anode_flow,
                "new_anode_setpoint": anode_flow_command,
            }

            data_log.append(sample)

            with open(out_file, "wb") as fd:
                pickle.dump(data_log, fd)

            print(f"PID data written to {out_file}")

            if args.verbose:
                print(
                    f"t={elapsed_time:7.2f} s | "
                    f"target={args.target_current:8.3f} A | "
                    f"measured={measured_current:8.3f} A | "
                    f"error={error:8.3f} A | "
                    f"flow_cmd={anode_flow_command*1e6:8.3f} mg/s"
                )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)



