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

import sys
sys.path.append("..")
from source.ocr import OCR
from source.models import GFCN
from source.ocr_dataset import OCRDataset
from torch.optim import Adam


if __name__ == "__main__":

    params = {
        "dataset_path": "../Datasets/iam_lines/iam_lines_64H_NKR",
        "output_folder": "GFCN_iam_lignes_64H_NKR",
        "set_names": ["train", "valid", "test"],
        "train_set_name": "train",
        "valid_set_names": ["valid", ],
        "test_set_names": ["test", ],
        "models": {
            "end_to_end_model": GFCN,
        },
        "augmentation": {
            "activated": False,
        },
        "normalization": {
            "type": "instance",
            "eps": 0.001,
            "momentum": 0.99,
            "num_group": 32,
            "track_running_stats": False,
        },
        "dropout": 0.4,  # dropout probability
        "height": 64,  # image height
        "nb_epochs": 1000,
        "load_epoch": "best",  # ["best", "last"]
        "batch_size": 2,
        "lr": 0.0001,
        "optimizer": Adam,
        "eval_on_valid": True,  # whether to compute metrics on valid set during training or not
        "eval_on_valid_interval": 2,  # epochs interval
        "input_channels": 1,  # 1 for grayscaled image, 3 for RGB
        "nb_gate": 6,  # number of GateBlocks to use in the GFCN model
        "reduce_len": 4,  # width division through model
        "dataset_class": OCRDataset,
        "losses": ["loss_ctc"],
        "metrics": ["edit"],
    }

    model = OCR(params)

    model.load_dataset()
    model.load_model()

    # model.train()

    # model.predict enable to compute metrics on a specific set
    # available sets: "train", "valid", "test"
    # available metrics: ["cer", "wer", "ground_truth", "pred", "proba", "time", "loss_ctc"]
    model.predict("test", ["cer", "wer"])
