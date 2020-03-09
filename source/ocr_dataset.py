from torch.utils.data import Dataset
import pickle
import os
import re
import h5py
import numpy as np
import random
from source.utils import pad_sequences_1d, pad_sequences_2d


class OCRDataset(Dataset):

    def __init__(self, root_folder, set_names=("train", "valid", "test")):
        self.root_folder = root_folder
        self.files = {}
        for set_name in set_names:
            self.files[set_name] = [os.path.join(root_folder, set_name, name) for name in os.listdir(os.path.join(root_folder, set_name))] if os.path.isdir(os.path.join(root_folder, set_name)) else None

        with open(os.path.join(root_folder, "params.pkl"), "rb") as f:
            self.params = pickle.load(f)

        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [atoi(c) for c in re.split('(\d+)', text)]

        for set_name in set_names:
            self.files[set_name].sort(key=natural_keys)

        self.reduce_len = None
        self.nb_ignored = {
            "train": 0,
            "valid": 0,
            "test": 0
        }

    def get_generator(self, set_, batch_size, shuffle=True, augmentation=False):
        files = self.files[set_]
        if shuffle:
            random.shuffle(files)
        iter_files = iter(files)
        self.nb_ignored[set_] = 0
        is_ignored_computed = False
        while True:
            data = []
            data_len = []
            data_reduced_len = []
            label = []
            label_len = []
            label_set = []
            img_name = []
            end_epoch = False
            while len(data) != batch_size:
                try:
                    file = next(iter_files)
                except StopIteration:
                    is_ignored_computed = True
                    end_epoch = True
                    if shuffle:
                        random.shuffle(files)
                    iter_files = iter(files)
                    break

                file = h5py.File(file, 'r')
                mdata = np.array(file["data"])
                mdata_len = file.attrs["data_len"]
                mdata_reduced_len = file.attrs["data_len"]
                mlabel = file.attrs["label_ind"]
                mlabel_len = file.attrs["label_len"]
                mlabel_set = 0 if file.attrs["dataset"] == "iam" else 1
                mimg_name = file.attrs["name"]
                file.close()

                if self.reduce_len is not None:
                    mdata_reduced_len = np.floor(mdata_reduced_len / self.reduce_len)

                if 2 * mlabel_len + 1 > mdata_reduced_len:
                    if not is_ignored_computed:
                        self.nb_ignored[set_] += mlabel_len
                    continue

                data.append(mdata)
                data_len.append(mdata_len)
                data_reduced_len.append(mdata_reduced_len)
                label.append(mlabel)
                label_len.append(mlabel_len)
                label_set.append(mlabel_set)
                img_name.append(mimg_name)

            # if set == "train" and augmentation["activated"]:
            #         add augmentation on the fly

            if augmentation["activated"]:
                # augmentation is performed on non-normalized data
                data = [(d - self.params["mean"]) / self.params["std"] for d in data]  #  normalization

            if len(data) > 0:
                data = pad_sequences_2d(data, padding_value=float(self.params["padding_value"]))
                label = pad_sequences_1d(label, padding_value=float(self.params["padding_value"])).astype(np.int32)

            yield [data, np.array(data_len, dtype=np.int32), np.array(data_reduced_len, dtype=np.int32), label,
                   np.array(label_len), np.array(label_set), img_name], end_epoch
