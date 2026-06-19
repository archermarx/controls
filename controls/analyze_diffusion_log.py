import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("logfile", type=str)
args = parser.parse_args()

with open(args.logfile, "rb") as fd:
    log = pickle.load(fd)


print(log.keys())
metadata = log["metadata"]
print(metadata)

lb = np.array(metadata["control_lb"])
ub = np.array(metadata["control_ub"])

iters = log["iterations"]
print(f"{iters[0]["step_scale"]=}")
stages = iters[0]["stages"]
print(f"{stages.keys()=}")

model_control = stages["get_model_proposed_control"]
print(model_control.keys())

c_proposed = np.array(model_control["c_proposed"])
z_proposed = np.array(model_control["z_proposed"])
inds = np.argsort(z_proposed)
z_proposed = z_proposed[inds]
c_proposed = c_proposed[inds]

print("c\tz\n--------------")


def in_bounds(c, lb, ub):
    return np.all(c >= lb) and np.all(c <= ub)


numerator = np.zeros(len(c_proposed[0]))
denominator = 0.0
for c, z in zip(c_proposed, z_proposed):
    print(f"{c[0]:.3f}\t{z:.3f}")

    if in_bounds(c, lb, ub):
        numerator += c / z**2
        denominator += 1 / z**2
    else:
        print("out of bounds")

print(f"Final control: {numerator / denominator}")
