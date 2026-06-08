import os
import pickle
import sys

import lib.labview as labview

dir = sys.argv[1]
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1)

for file in os.listdir(dir):
    with open(os.path.join(dir, file), "rb") as fd:
        contents = pickle.load(fd)

    oscope: labview.OscopeReadings = contents["data"]["oscope"]["Anode Current"]
    print(contents["data"]["oscope"].keys())
    rms = oscope.rms
    average = oscope.average
    peak_to_peak = oscope.peak_to_peak
    waveform = oscope.waveform

    dmm = contents["data"]["dmm"].current
    print(f"Average (DMM): {dmm} A")
    print(f"Average (oscope): {average} A")
    print(f"Peak to peak (oscope): {peak_to_peak} A")

    t, I = waveform.time_values(), waveform.y_values()
    if average < 12:
        print(len(I))
        print(np.mean(I))

    t = t * 1000
    ax.set_xlim(0, 0.5)
    ax.plot(t, I)

fig.savefig("oscillations.png")
