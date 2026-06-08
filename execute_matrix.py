import argparse
import math
import os
from pathlib import Path
import pickle

import pandas as pd

import lib.controls as controls
import lib.labview as labview

parser = argparse.ArgumentParser()
parser.add_argument("test_matrix", type=Path, help="CSV file containing test points")
parser.add_argument("--coil-currents", "-c", type=float, nargs=2, help="Inner and outer magnet coil currents at the 100 percent nominal b-field setting")
parser.add_argument("--data", "-d", type=lambda s: s.split(","), help="Comma-separated list of data types to collect. Choices are 'magna', 'alicat', 'lambda', 'dmm', and 'oscope'. Defaults to writing all.")
parser.add_argument("--output", "-o", type=Path, default=Path("."), help="Folder in which data will be written. Will be created if it does not already exist.")
parser.add_argument("--prefix", "-p", type=str, default="data", help="Prefix to append to data files.")
parser.add_argument("--gas", "-g", type=str, choices=["Xe", "Kr", "Ar"], default="Kr", help="Propellant gas being used. One of 'Xe', 'Kr', or 'Ar'. Defaults to 'Kr'.")
parser.add_argument("--verbose", "-v", action="store_true", help="Whether to print the raw bytestrings sent to LabVIEW.")
parser.add_argument("--dwell-time", "-t", type=int, default=5, help="How long (in seconds) to dwell at each operating point before collecting data.")
parser.add_argument("--host-ip", type=str, default="169.254.144.78", help="The IP address of the LabVIEW client")
parser.add_argument("--port", type=int, default=59704, help="The port of the LabVIEW client")
parser.add_argument("--interactive", "-i", action="store_true", help="Whether to ask the user before proceeding to the next control point")

def num_digits(n):
    if n > 0:
        digits = int(math.log10(n))+1
    elif n == 0:
        digits = 1
    else:
        digits = int(math.log10(-n))+2 # +1 if you don't count the '-' 
    return digits

def main(args):
    inner_coil_current, outer_coil_current = args.coil_currents
    prefix = "" if not args.prefix else args.prefix + "_"
    data_types = ["magna", "dmm", "alicat", "lambda", "oscope"] if args.data is None else args.data

    matrix = pd.read_csv(args.test_matrix)
    flow_rates = matrix["anode_flow_rate_kg_s"]
    discharge_voltages = matrix["discharge_voltage_v"]
    cathode_flow_fractions = matrix["cathode_flow_fraction"]
    magnetic_field_strengths = matrix["magnetic_field_scale"]

    controller = controls.ThrusterController(args.gas, args.verbose)
    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)

    num_elems = len(flow_rates)
    ndigits = num_digits(num_elems)
    filename_format = prefix + f"{{:0{ndigits}d}}.pkl"

    with labview.LabViewClient(host=args.host_ip, port=args.port) as client:
        for (i, (mdot, vd, cff, bmag)) in enumerate(zip(flow_rates, discharge_voltages, cathode_flow_fractions, magnetic_field_strengths)):

            setpoint = controls.ControlPoint(
                anode_flow_rate_kg_s=mdot,
                discharge_voltage_V=vd,
                cathode_flow_fraction=cff,
                magnet_current_inner_A=bmag*inner_coil_current,
                magnet_current_outer_A=bmag*outer_coil_current,
            )
            
            print(f"Setpoint {i+1}: {setpoint}")
            controller.control_to(setpoint, client)
            data = controller.take_data(client, delay=args.dwell_time, sources=data_types)
            out_file = output_dir / filename_format.format(i+1)

            output = {
                "controls": setpoint.model_dump(),
                "data": data,
            }

            with open(out_file, "wb") as fd:
                pickle.dump(output, fd)

            print(f"Setpoint {i+1}: data written to {out_file}")

            if args.interactive:
                while True:
                    answer = input("Proceed to next point (y/n)? ")
                    if answer.casefold() == "y":
                        break

if __name__ == "__main__":
    args = parser.parse_args()
    if args.coil_currents is None:
        raise ValueError("No coil currents specified!")


    main(args)
