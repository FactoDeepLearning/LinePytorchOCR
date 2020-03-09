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
