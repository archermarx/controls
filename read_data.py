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
    rms = oscope.rms
    average = oscope.average
    peak_to_peak = oscope.peak_to_peak
    waveform = oscope.waveform

    print(f"{peak_to_peak=}")
    print(f"{average=}")
    t, I = waveform.time_values(), waveform.y_values()
    t = t * 1000
    ax.set_xlim(0, 2.0)
    ax.plot(t, I)

fig.savefig("oscillations.png")
