# PytorchOCR
A pytorch implementation of a line optical character recognition (OCR).

We provide a pre-formatted line IAM dataset and a pre-trained Gated Fully Convolutional (GFCN) model reaching 5.23% of CER for the validation set and 7.99% for the test set.

## Format your data
Your dataset folder must be placed in the "Datasets" folder and be structured in the following way:
* Datasets
    * MyDataset
        * train
            * train_images
        * valid
            * valid_images
        * test
            * test_images
        * train.txt
        * valid.txt
        * test.txt
        
where train, valid and test are folders containing the text line images and train.txt, valid.txt and test.txt and the 
files containing the corresponding ground truths.

Those files should be formatted like that: <br/>
path/img_name.extension ground_truth <br/>
path/img_name2.extension ground_truth_2 <br/>
...


*Exemple: train.txt*

train/img1.png this is my ground truth <br/>
train/img2.png another ground truth <br/>


Folder and file names can be personnalised in the format_data.py file
A ready-to-use version of the IAM lines dataset is given as example in the root folder:

```bash
tar -xzf iam_lines.tar.gz -C Datasets/
```

You can then format the dataseet running *source/format_data.py*


## Train your model
Pytorch models are defined in *source/models.py*.
As for the provided GFCN example, its forward function must return the log_softmax values for each frame.

Your model must be specified in *source/main.py* as well as the desired training parameters. Then, call model.train().

Training your model generates some files:
* In *source/outputs/model_name/results*:
    * events files (for tensorboard visualization)
    * *params.txt* which contains the used training parameters
    * *summary_model.txt* which contains a description of the parameters involved in the model 
    and which show the evolution of the receptive field through the layers
* In *source/outputs/model_name/checkpoints*:
    * model weights (.pt) for the epoch reaching the best CER (for validation set) and for the last completed epoch.

## Evaluate your model
Run *source/models" with the same parameters you use for training and call model.predict("test", ["cer", "wer"]).

The first argument is the set to use: "train", "valid" or "test"

The second argument is a list of metrics among ["cer", "wer", "ground_truth", "pred", "proba", "time", "loss_ctc"].

Computed metrics are then written in text files in *source/outputs/model_name/results*.

