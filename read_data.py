import os
import pickle
import sys

import lib.labview as labview

dir = sys.argv[1]
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
    print(f"{waveform=}")


fig.savefig("oscillations.png")
