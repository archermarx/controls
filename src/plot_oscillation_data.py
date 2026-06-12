import os
import pickle
import argparse

import lib.labview as labview
import lib.controls as controls

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("dir", type=str, help="Directory in which data files are stored")
parser.add_argument("--output-dir", "-o", type=str, help="Output directory")

args = parser.parse_args()
dir = args.dir

output_dir = args.output_dir if args.output_dir else dir

fig, ax = plt.subplots(1,1)
bfields = []
mdots = []
voltages = []
cffs = []
p2ps = []
rmss = []
avgs = []

for file in os.listdir(dir):
    if os.path.isdir(file):
        continue
    elif os.path.splitext(file)[1] != ".pkl":
        continue

    with open(os.path.join(dir, file), "rb") as fd:
        contents = pickle.load(fd)

    oscope: labview.OscopeReadings = contents["data"]["oscope"]["Anode Current"]

    setpoint = contents["controls"]
    rms = oscope.rms
    average = oscope.average
    peak_to_peak = oscope.peak_to_peak
    waveform = oscope.waveform

    dmm = contents["data"]["dmm"].current
    print(f"Average (DMM): {dmm} A")
    print(f"Average (oscope): {average} A")
    print(f"Peak to peak (oscope): {peak_to_peak} A")

    if peak_to_peak > 1e3:
        print(f"Oscope clipped for {setpoint}. Skipping.\n")
        continue

    if len(oscope.waveform.data) == 0:
        print(f"Warning: no oscillation data collected for {setpoint}. Skipping.\n")
        continue

    bfield_percent = setpoint["magnetic_field_scale"]
    voltage = setpoint["discharge_voltage_v"]
    anode_flow = setpoint["anode_mass_flow_rate_kg_s"]
    cff = setpoint["cathode_flow_fraction"]
    voltages.append(voltage)
    mdots.append(anode_flow)
    cffs.append(cff)
    bfields.append(bfield_percent)
    p2ps.append(peak_to_peak)
    avgs.append(dmm)

    t, I = waveform.time_values(), waveform.y_values()

    mean_I = np.mean(I)
    rms = np.sqrt(np.mean((I - mean_I)**2))

    rmss.append(rms)

    t = t * 1000
    ax.set_xlim(0, 0.5)
    ax.plot(t, I)
    print()

plot_dir = os.path.join(output_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

fig.savefig(os.path.join(plot_dir, "oscillations.png"))

xvars = [bfields, mdots, cffs, voltages]
labels = ["bfield", "mdot", "cff", "voltage"]
titles = ["Fraction of max. B-field", "Anode flow rate (kg/s)", "Cathode flow fraction", "Discharge Voltage"]
for var, label, title in zip(xvars, labels, titles):
    fig, axs = plt.subplots(3,1, figsize=(6,6), layout='constrained')
    sort_inds = np.argsort(var)
    var_sorted = np.array(var)[sort_inds]
    avgs_sorted = np.array(avgs)[sort_inds]
    rmss_sorted = np.array(rmss)[sort_inds]
    p2ps_sorted = np.array(p2ps)[sort_inds]

    axs[0].set(ylabel = "Amperes", title = "Average discharge current", xticklabels = [])
    axs[0].plot(var_sorted, avgs_sorted, '-o')

    axs[1].plot(var_sorted, rmss_sorted / avgs_sorted * 100, '-o')
    axs[1].set(ylabel = "Amplitude (%)", title = "RMS Amplitude (%)", xticklabels = [])
    axs[1].set_ylim(bottom=0.0)

    axs[2].scatter(var_sorted, p2ps_sorted / avgs_sorted * 100)
    axs[2].set(xlabel = title, ylabel = "Amplitude (%)", title = "Peak to peak (%)")
    axs[2].set_ylim(bottom=0.0)

    fig.savefig(os.path.join(plot_dir, f"{label}_sweep.png"), dpi=200)