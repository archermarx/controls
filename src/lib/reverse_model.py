import copy
import tomllib

from pathlib import Path

import numpy as np
import torch

from hall_diffusion import sample as sampling
import hall_diffusion.utils.thruster_data as thruster_data


class ReverseModel:
    def __init__(
        self,
        model: str | Path,
        config: str | Path | dict,
        sample_dir: str | Path,
        replace_samples: bool = False,
        verbose: bool = False,
    ):
        self.model = Path(model)
        self.sample_dir = Path(sample_dir)
        self.replace_samples = replace_samples
        self.verbose = verbose

        # Read config file and extract some useful information
        if isinstance(config, Path) or isinstance(config, str):
            with open(config, "rb") as fd:
                self.config = tomllib.load(fd)
        else:
            self.config = config

        self.base_sim = self.config["observation"]["base_sim"]
        self.stddev = self.config["observation"]["stddev"]
        self.config.pop("observation")

        self.dataset = thruster_data.ThrusterDataset(
            self.base_sim, scalars_in_tensor=True, fourier_features=True
        )

    def build_config(self, data, num_samples, num_steps):
        # TODO: incorporate data + design struct for data
        config = copy.deepcopy(self.config)

        observation = {
            "base_sim": self.base_sim,
            "stddev": self.stddev,
            "fields": {},
        }

        for key, val in data.items():
            if (
                key in self.dataset.params()
                or key in self.dataset.norm.norm_perf["names"]
            ):
                if isinstance(val, dict):
                    mean = val["mean"]
                    std = val.get("std", self.stddev)
                else:
                    mean = val
                    std = self.stddev

                observation["fields"][key] = {
                    "x": "all",
                    "y": [mean],
                    "stddev": std,
                    "normalized": False,
                }
            elif key in self.dataset.fields():
                observation["fields"][key] = val

            else:
                print(f"Skipped key {key} in conditioning diffusion model.")
                continue

        config["observation"] = observation
        config["out_dir"] = str(self.sample_dir)
        config["num_samples"] = num_samples
        config["num_steps"] = num_steps
        return config

    def get_scalar_params(self, x):
        if len(x.shape) == 2:
            x = x[None, ...]

        params = {}
        for k in self.dataset.params():
            c_ki = self.dataset.get_field(x, k)
            params[k] = np.mean(c_ki, axis=1)

        return params

    def __call__(self, data, num_samples, num_steps, save_to_file=True):
        # Runs the diffusion model
        # Outputs samples to sample_dir
        # Loads samples, returns for use by forward model
        config = self.build_config(data, num_samples, num_steps)

        if "discharge_current_signal" in data:
            Id_dict = data.pop("discharge_current_signal")
            t_vec = torch.tensor(Id_dict["time"])
            I_vec = torch.tensor(Id_dict["current"])
            condition_vec = self.dataset._signal_to_vec(t_vec, I_vec, truncate=False)
        else:
            condition_vec = None

        samples_allsteps = sampling.infer(
            self.model,
            config,
            scalars_in_tensor=True,
            fourier_features=True,
            save_to_file=save_to_file,
            verbose=self.verbose,
            condition_vec=condition_vec,
        )
        samples = samples_allsteps[-1, ...]

        state_ests = self.dataset.norm.denormalize_tensor(samples)
        return state_ests
