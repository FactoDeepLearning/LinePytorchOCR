#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#
#  This software is a computer program written in XXX whose purpose is XXX.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

import os
from PIL import Image
import numpy as np
import h5py
import pickle


class DatasetFormatter():

    def __init__(self, dataset_info, format_info):
        self.dataset_info = dataset_info
        self.format_info = format_info
        self.output_path = None
        self.img_paths = {}
        for key in self.dataset_info["img_folders"].keys():
            self.img_paths[key] = os.listdir(os.path.join(self.dataset_info["root_path"], self.dataset_info["img_folders"][key]))
            self.img_paths[key] = [os.path.join(self.dataset_info["root_path"], self.dataset_info["img_folders"][key], name) for name in self.img_paths[key]]

    def compute_std_mean(self, nb_channels):
        sum = np.zeros((nb_channels, ))
        nb_pixels = 0
        for key in ("train", ):
            files = os.listdir(os.path.join(self.output_path, key))
            for file in [os.path.join(self.output_path, key, name) for name in files]:
                file = h5py.File(file, 'r')
                img = np.array(file["data"])
                sum += np.sum(img, axis=(0, 1))
                nb_pixels += np.prod(img.shape[:2])
                file.close()
        self.dataset_info["mean"] = sum / nb_pixels
        diff = np.zeros((nb_channels, ))
        for key in ("train", ):
            files = os.listdir(os.path.join(self.output_path, key))
            for file in [os.path.join(self.output_path, key, name) for name in files]:
                file = h5py.File(file, 'r')
                img = np.array(file["data"])
                diff += [np.sum((img[:, :, c]-self.dataset_info["mean"][c])**2) for c in range(nb_channels)]
                file.close()
        self.dataset_info["std"] = np.sqrt(diff/nb_pixels)

    def compute_labels_info(self):
        labels = []
        for key in self.dataset_info["img_folders"].keys():
            file_path = os.path.join(self.dataset_info["root_path"], self.dataset_info["label_files"][key])
            nb_char = 0
            with open(file_path, "r") as f:
                for line in f:
                    ground_truth = " ".join(line.split(" ")[1:]).strip()
                    nb_char += len(ground_truth)
                    for c in ground_truth:
                        if c not in labels and c != "\n":
                            labels.append(c)
            self.dataset_info["nb_char"][key] = nb_char
        if not self.dataset_info["labels"]:
            self.dataset_info["labels"] = sorted(labels)

    def save_params(self, path):
        with open(os.path.join(path, "params.txt"), "w") as f:
            for key in self.dataset_info.keys():
                value = self.dataset_info[key]
                if callable(value):
                    value = value.__name__
                f.write("{}: {}\n".format(key, value))
        with open(os.path.join(path, "params.pkl"), "wb") as f:
            pickle.dump({
                "labels": self.dataset_info["labels"],
                "nb_char": self.dataset_info["nb_char"],
                "std": self.dataset_info["std"],
                "mean": self.dataset_info["mean"],
                "padding_value": self.format_info["padding_value"]
            }, f)

    def str_to_indices(self, label):
        res = []
        for c in label:
            try:
                res.append(self.dataset_info["labels"].index(c))
            except ValueError:
                print("Warning - char not in charset: "+c+"\n")
        return np.array(res)

    def format(self):
        output_fold_name = self.dataset_info["root_path"].split("/")[-1]
        if self.format_info["height"]:
            output_fold_name += "_{}H".format(self.format_info["height"])
            output_fold_name += "_KR" if self.format_info["keep_ratio"] else "_NKR"
        else:
            output_fold_name += "_RAW"
        if not self.format_info["normalize"]:
            output_fold_name += "_NN"
        output_fold_name += self.format_info["output_name_extra"]
        self.output_path = os.path.join(self.dataset_info["root_path"], output_fold_name)
        os.makedirs(self.output_path, exist_ok=True)

        self.compute_labels_info()

        print("resizing...")
        #first-pass : create h5py files from image and ground truth + resize
        for key in self.dataset_info["img_folders"].keys():
            fold_path = os.path.join(self.output_path, key)
            os.makedirs(fold_path)
            file_path = os.path.join(self.dataset_info["root_path"], self.dataset_info["label_files"][key])
            with open(file_path, "r") as f:
                for i, line in enumerate(f):
                    gt = " ".join(line.split(" ")[1:]).strip()
                    img_name = line.split(" ")[0]
                    img_path = os.path.join(self.dataset_info["root_path"], self.dataset_info["label_path_dirs"][key], img_name)
                    img = Image.open(img_path)

                    img_output_path = os.path.join(fold_path, "{}.hdf5".format(i))

                    img = np.array(img)

                    if self.format_info["height"]:
                        shape = img.shape
                        h, w = shape[:2]
                        c = shape[2] if len(shape) == 3 else 1

                        if self.format_info["keep_ratio"]:
                            ratio = self.format_info["height"] / h
                            img = np.array(Image.fromarray(img).resize((int(w * ratio), self.format_info["height"]), Image.ANTIALIAS))
                        else:
                            img = np.array(Image.fromarray(img).resize((w, self.format_info["height"]), Image.ANTIALIAS))

                    if len(img.shape) == 2:
                        img = np.expand_dims(img, 2)

                    out_file = h5py.File(img_output_path, "w")
                    out_file.create_dataset("data", data=img, dtype="float32", compression="gzip", compression_opts=5)
                    out_file.attrs["label_ind"] = self.str_to_indices(gt)

                    out_file.attrs["label"] = gt
                    out_file.attrs["dataset"] = self.format_info["dataset_name"]
                    out_file.attrs["label_len"] = len(gt)
                    out_file.attrs["name"] = img_name
                    out_file.close()

        if not self.dataset_info["mean"] or not self.dataset_info["std"]:
            print("computing mean & std...")
            self.compute_std_mean(nb_channels=c)

        print("normalizing...")
        # second-pass : compute mean/std, normalize
        for key in self.dataset_info["img_folders"].keys():
            files = os.listdir(os.path.join(self.output_path, key))
            for file in [os.path.join(self.output_path, key, name) for name in files]:
                file = h5py.File(file, 'r+')
                img = np.array(file["data"])

                if self.format_info["normalize"]:
                    for i in range(c):
                        img[:, :, i] = (img[:, :, i] - self.dataset_info["mean"][i]) / self.dataset_info["std"][i]

                h, w, c = img.shape

                file["data"][:] = img
                file.attrs["data_len"] = w
                file.close()
        self.save_params(self.output_path)


if __name__ == "__main__":

    dataset_info = {
        "root_path": "../Datasets/iam_lines",

        "img_folders": {
            "train": "train",
            "valid": "valid",
            "test": "test"
        },

        "label_files": {
            "train": "train.txt",
            "valid": "valid.txt",
            "test": "test.txt"
        },

        "label_path_dirs": {
            "train": "",
            "valid": "",
            "test": ""
        },

        "std": None,
        "mean": None,
        "labels": [],
        "nb_char": {},
    }

    format_info = {
        "height": 64,
        "keep_ratio": False,
        "normalize": True,
        "dataset_name": "iam",
        "padding_value": 300,
        "output_name_extra": ""
    }

    formatter = DatasetFormatter(dataset_info, format_info)
    formatter.format()