import os
import numpy as np
import glob
import json
from collections.abc import Sequence

from .builder import DATASETS
from .defaults import DefaultDataset

from pointcept.utils.cache import shared_dict

@DATASETS.register_module()
class So100Dataset(DefaultDataset):

    def get_data_list(self):
        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = self.split
        else:
            raise NotImplementedError

        data_list = []
        for split in split_list:
            path = os.path.join(self.data_root, split)
            # if split is a file listing JSON, keep compatibility
            if os.path.isfile(path):
                with open(path) as f:
                    data_list += [os.path.join(self.data_root, d) for d in json.load(f)]
            # if a directory, gather .npy files
            elif os.path.isdir(path):
                data_list += glob.glob(os.path.join(path, "*.npy"))
            else:
                # allow glob patterns or direct file paths
                data_list += glob.glob(os.path.join(self.data_root, split))
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        raw = np.load(data_path, allow_pickle=True)
        # handle files saved via np.save(dict)
        if isinstance(raw, np.ndarray) and raw.dtype == object:
            try:
                data_src = raw.item()
            except Exception:
                # fall back to raw array
                data_src = {"coord": raw}
        elif isinstance(raw, dict):
            data_src = raw
        else:
            # unexpected format: try to interpret as coord array
            data_src = {"coord": raw}

        data_dict = {}
        # copy recognized assets
        for k in ["coord", "color", "normal", "strength", "segment", "instance", "pose"]:
            if k in data_src:
                data_dict[k] = data_src[k]

        data_dict["name"] = name
        data_dict["split"] = split

        if "coord" in data_dict:
            data_dict["coord"] = data_dict["coord"].astype(np.float32)

        if "color" in data_dict:
            data_dict["color"] = data_dict["color"].astype(np.float32)

        if "normal" in data_dict:
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "segment" in data_dict:
            data_dict["segment"] = data_dict["segment"].reshape([-1]).astype(np.int32)
        else:
            data_dict["segment"] = np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1

        if "instance" in data_dict:
            data_dict["instance"] = data_dict["instance"].reshape([-1]).astype(np.int32)
        else:
            data_dict["instance"] = np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1

        return data_dict

    def get_data_name(self, idx):
        path = self.data_list[idx % len(self.data_list)]
        return os.path.splitext(os.path.basename(path))[0]

    def get_split_name(self, idx):
        path = self.data_list[idx % len(self.data_list)]
        return os.path.basename(os.path.dirname(path))