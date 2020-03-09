import os
import sys
from time import time

import numpy as np
import torch
from torch.nn import CTCLoss, Conv2d, Linear
from torch.nn.init import xavier_uniform_, zeros_
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import editdistance
from contextlib import redirect_stdout

from source.utils import model_summary, edit_cer_from_list, edit_wer_from_list, receptive_field


class OCR:

    def __init__(self, params):
        self.params = params
        self.all_labels = []
        self.dataset = None
        self.models = {}
        self.latest_epoch = -1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda:0":
            sys.stderr.write("{} {}\n".format(torch.cuda.get_device_name(), torch.cuda.get_device_properties(self.device)))
        self.params["device"] = self.device
        self.optimizer = None
        self.best = None

        output_path = os.path.join("outputs", self.params["output_folder"])
        os.makedirs(output_path, exist_ok=True)
        checkpoints_path = os.path.join(output_path, "checkpoints")
        os.makedirs(checkpoints_path, exist_ok=True)
        results_path = os.path.join(output_path, "results")
        os.makedirs(results_path, exist_ok=True)

        self.paths = {
            "results": results_path,
            "checkpoints": checkpoints_path,
            "output_folder": output_path
        }

        self.writer = SummaryWriter(self.paths["results"])

    def load_dataset(self):
        self.dataset = self.params["dataset_class"](os.path.join(self.params['dataset_path']), self.params["set_names"])
        self.all_labels = self.dataset.params["labels"].copy()
        self.dataset.reduce_len = self.params["reduce_len"]
        self.params["vocab_size"] = len(self.all_labels)
        self.params["padding_value"] = self.dataset.params["padding_value"]

    def ctc_generator_to_values(self, batch_data):
        batch_imgs = batch_data[0]
        batch_imgs_len = batch_data[1]
        batch_imgs_reduced_len = batch_data[2]
        batch_labels = batch_data[3]
        batch_labels_len = batch_data[4]
        batch_set = batch_data[5]
        batch_img_name = batch_data[6]
        return batch_imgs, batch_labels, batch_imgs_len, batch_imgs_reduced_len, batch_labels_len, batch_set, batch_img_name

    @staticmethod
    def weights_init(m):
        if isinstance(m, Conv2d) or isinstance(m, Linear):
            xavier_uniform_(m.weight)
            zeros_(m.bias)

    def load_model(self):
        for model_name in self.params["models"].keys():
            self.models[model_name] = self.params["models"][model_name](self.params)

        checkpoint_path = None
        checkpoint = None

        for filename in os.listdir(self.paths["checkpoints"]):
            if self.params["load_epoch"] in filename:
                checkpoint_path = os.path.join(self.paths["checkpoints"], filename)
                break
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.latest_epoch = checkpoint["epoch"]
            self.best = checkpoint["best"]
            for model_name in self.models.keys():
                self.models[model_name].load_state_dict(checkpoint["{}_state_dict".format(model_name)])

        for model_name in self.models.keys():
            self.models[model_name].to(self.device)

        parameters = list()
        for model_name in self.models.keys():
            parameters += list(self.models[model_name].parameters())

        self.optimizer = self.params["optimizer"](parameters, lr=self.params["lr"])
        if checkpoint_path:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            for model_name in self.models.keys():
                self.models[model_name].apply(self.weights_init)

    def save_model(self, epoch, name):
        for filename in os.listdir(self.paths["checkpoints"]):
            if name in filename:
                os.remove(os.path.join(self.paths["checkpoints"], filename))
        path = os.path.join(self.paths["checkpoints"], "{}_{}.pt".format(name, epoch))
        content = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'best': self.best,
        }
        for model_name in self.models.keys():
            content["{}_state_dict".format(model_name)] = self.models[model_name].state_dict()
        torch.save(content, path)

    def save_summary(self):
        """
        Save description of the model into txt file
        """
        path = os.path.join(self.paths["results"], "summary_model.txt")
        try:
            with open(path, 'w') as f:
                with redirect_stdout(f):
                    model_summary(self.models["end_to_end_model"], (1, self.params["input_channels"], self.params["height"], 1000), self.device)
                    receptive_field(self.models["end_to_end_model"], (2, self.params["input_channels"], self.params["height"], 1000), self.device)
        except Exception as e:
            print(e)

    def batch_cer(self, truth, pred):
        cer = 0
        for t, p in zip(truth, pred):
            cer += editdistance.eval(t, p) / len(t)
        return cer / len(truth)

    def batch_edit(self, truth, pred):
        edit = 0
        for t, p in zip(truth, pred):
            edit += editdistance.eval(t, p)
        return edit

    def batch_len(self, truth, pred):
        return np.mean([abs(len(t) - len(p)) for t, p in zip(truth, pred)])

    @staticmethod
    def batch_probas_to_str(probas, names):
        res = []
        for proba, name in zip(probas, names):
            array_str = np.array2string(proba.T, threshold=np.inf, max_line_width=np.inf, formatter={'float_kind':lambda x: "%.6f" % x}).replace("[", "").replace("]", "")
            proba_str = "{} [\n{} ]".format(name, array_str)
            res.append(proba_str)
        return ["\n".join(res)]

    def save_params(self):
        with open(os.path.join(self.paths["results"], "params"), "w") as f:
            for key in self.params.keys():
                value = self.params[key]
                if callable(value):
                    value = value.__name__
                f.write("{}: {}\n".format(key, value))
            f.write("labels: {}".format(self.all_labels))
        self.save_summary()

    def ctc_probas_to_ind(self, probas):
        res = []
        for p in probas:
            res.append(torch.argmax(p, dim=0))
        return res

    def ctc_ind_to_str(self, ind):
        res = ""
        for i in ind:
            res += self.all_labels[i] if i < len(self.all_labels) else ""
        return res

    def ctc_str_to_ind(self, label):
        res = []
        for c in label:
            res.append(self.all_labels.index(c))
        return res

    @staticmethod
    def ctc_remove_successives_identical_ind(ind):
        res = []
        for i in ind:
            if res and res[-1] == i:
                continue
            res.append(i)
        return res

    def ctc_decode(self, probas):
        return self.ctc_ind_to_str(self.ctc_remove_successives_identical_ind(self.ctc_probas_to_ind(probas)))

    def train(self):
        self.save_params()
        nb_epochs = self.params["nb_epochs"]
        batch_size = self.params["batch_size"]
        nb_batch = int(np.ceil(len(self.dataset.files["train"]) / batch_size)) + 1
        train_generator = self.dataset.get_generator(self.params["train_set_name"], batch_size, augmentation=self.params["augmentation"])
        values = None
        for e in range(self.latest_epoch+1, nb_epochs):
            self.latest_epoch = e
            losses = {}
            for loss_name in self.params["losses"]:
                losses[loss_name] = 0
            metrics = {}
            for metric_name in self.params["metrics"]:
                metrics[metric_name] = 0
            t = tqdm(range(1, nb_batch + 1))
            t.set_description("EPOCH {}/{}".format(e, nb_epochs))
            for i in t:
                batch_data, is_end_epoch = next(train_generator)
                if isinstance(batch_data[0], list):
                    break
                batch_losses, batch_metrics = self.train_batch(self.ctc_generator_to_values(batch_data))
                for key in losses.keys():
                    losses[key] += batch_losses[key]
                for key in metrics.keys():
                    metrics[key] += batch_metrics[key]
                values = dict(losses)
                values.update(metrics)
                for key in values.keys():
                    values[key] = round(values[key] / i, 3)
                try:
                    del values["edit"]
                except KeyError:
                    pass
                t.set_postfix(values=str(values))
                if is_end_epoch:
                    break

            values["cer"] = round(metrics["edit"] /
                                  (self.dataset.params["nb_char"]["train"]-self.dataset.nb_ignored["train"]), 4)

            t.set_postfix(values=str(values))

            self.save_model(epoch=e, name="last")
            for key in values.keys():
                self.writer.add_scalar('train_{}'.format(key), values[key], e)
            if self.params["eval_on_valid"] and e % self.params["eval_on_valid_interval"] == 0:
                eval_values = None
                for valid_set_name in self.params["valid_set_names"]:
                    eval_values = self.evaluate(valid_set_name)
                    for key in eval_values.keys():
                        self.writer.add_scalar('{}_{}'.format(valid_set_name, key), eval_values[key], e)
                if self.best is None or eval_values["cer"] < self.best:
                    self.save_model(epoch=e, name="best")
                    self.best = eval_values["cer"]
            self.writer.flush()

    def train_batch(self, params):
        x, y, seq_len, seq_reduced_len, labels_len, _, _ = params
        x = torch.from_numpy(x).float().permute(0, 3, 1, 2).to(self.device)
        y = torch.from_numpy(y).long().to(self.device)

        for model_name in self.models.keys():
            self.models[model_name].train()

        loss_ctc = CTCLoss(blank=len(self.all_labels))
        self.optimizer.zero_grad()

        global_pred = self.models["end_to_end_model"](x)
        loss = loss_ctc(global_pred.permute(2, 0, 1), y, seq_reduced_len.tolist(), labels_len.tolist())
        loss_val = loss.item()
        loss.backward()
        self.optimizer.step()

        truth = [self.ctc_ind_to_str(i) for i in y]

        pred = [self.ctc_decode(pred) for pred in global_pred.permute(0, 2, 1)]

        edit = self.batch_edit(truth, pred)

        diff_len = self.batch_len(truth, pred)
        losses = {"loss_ctc": loss_val}
        metrics = {
            "edit": edit,
            "diff_len": diff_len
        }
        return losses, metrics

    def evaluate(self, set_name):
        batch_size = self.params["batch_size"]
        nb_batch = int(np.ceil(len(self.dataset.files[set_name]) / batch_size))
        generator = self.dataset.get_generator(set_name, batch_size, shuffle=False,
                                               augmentation=self.params["augmentation"])
        nb_char = self.dataset.params["nb_char"][set_name]
        values = None
        losses = {}
        for loss_name in self.params["losses"]:
            losses[loss_name] = 0
        metrics = {}
        for metric_name in self.params["metrics"]:
            metrics[metric_name] = 0
        t = tqdm(range(1, nb_batch + 1))
        t.set_description("Evaluation")
        for i in t:
            batch_data, is_end_epoch = next(generator)
            if isinstance(batch_data[0], list):
                break
            batch_losses, batch_metrics = self.evaluate_batch(self.ctc_generator_to_values(batch_data))
            for key in losses.keys():
                losses[key] += batch_losses[key]
            for key in metrics.keys():
                metrics[key] += batch_metrics[key]
            values = dict(losses)
            values.update(metrics)
            for key in values.keys():
                values[key] = round(values[key] / i, 4)
            try:
                del values["edit"]
            except KeyError:
                pass
            t.set_postfix(values=str(values))
            if is_end_epoch:
                break

        values["cer"] = round(metrics["edit"] / (nb_char - self.dataset.nb_ignored[set_name]), 4)
        t.set_postfix(values=str(values))
        return values

    def evaluate_batch(self, params):
        with torch.no_grad():
            x, y, seq_len, seq_reduced_len, labels_len, _, _ = params
            x = torch.from_numpy(x).float().permute(0, 3, 1, 2).to(self.device)
            y = torch.from_numpy(y).long().to(self.device)

            for model_name in self.models.keys():
                self.models[model_name].eval()
            loss_ctc = CTCLoss(blank=len(self.all_labels))

            global_pred = self.models["end_to_end_model"](x)

            loss = loss_ctc(global_pred.permute(2, 0, 1), y, seq_reduced_len.tolist(), labels_len.tolist())
            loss_val = loss.item()

        truth = [self.ctc_ind_to_str(i) for i in y]

        pred = [self.ctc_decode(pred) for pred in global_pred.permute(0, 2, 1)]

        edit = self.batch_edit(truth, pred)

        diff_len = self.batch_len(truth, pred)
        losses = {"loss_ctc": loss_val}
        metrics = {
            "edit": edit,
            "diff_len": diff_len
        }
        return losses, metrics

    def predict(self, set_name="test", metrics_name=["cer"]):
        batch_size = self.params["batch_size"]

        nb_batch = int(np.ceil(len(self.dataset.files[set_name]) / batch_size))
        generator = self.dataset.get_generator(set_name, batch_size, shuffle=False,
                                               augmentation=self.params["augmentation"])
        nb_char = self.dataset.params["nb_char"][set_name]
        metrics = {}
        for metric_name in metrics_name:
            if metric_name in ("pred", "ground_truth", "proba"):
                metrics[metric_name] = []
            else:
                metrics[metric_name] = 0
                if metric_name == "wer":
                    metrics["nb_words"] = 0

        if list(metrics.keys()) != ["time", ]:
            t = tqdm(range(1, nb_batch + 1))
            t.set_description("Prediction")
            for i in t:
                batch_data, is_end_epoch = next(generator)
                if isinstance(batch_data[0], list):
                    break
                batch_metrics = self.predict_batch(self.ctc_generator_to_values(batch_data), metrics_name)
                for key in batch_metrics.keys():
                    metrics[key] += batch_metrics[key]
                if is_end_epoch:
                    break
            values = dict(metrics)
            for key in metrics.keys():
                if key not in ("pred", "ground_truth", "proba"):
                    values[key] = round(values[key] / i, 4)
            if "cer" in values.keys():
                values["cer"] = round(metrics["cer"] / (nb_char - self.dataset.nb_ignored[set_name]), 4)
            if "wer" in values.keys():
                values["wer"] = round(metrics["wer"] / metrics["nb_words"], 4)
        else:
            values = dict(metrics)

        if "time" in metrics_name:
            n = 2
            t = tqdm(range(1, 1+n))
            t.set_description("Time")
            t_begin = time()
            for i in t:
                generator = self.dataset.get_generator(set_name, batch_size, shuffle=False, augmentation=self.params["augmentation"])
                is_end_epoch = False
                while not is_end_epoch:
                    batch_data, is_end_epoch = next(generator)
                    if not isinstance(batch_data[0], list):
                        _ = self.predict_batch(self.ctc_generator_to_values(batch_data), [])
            values["time-epoch"] = (time() - t_begin) / n * 1000
            values["time-char"] = (time()-t_begin) / n / (nb_char - self.dataset.nb_ignored[set_name]) * 1000
            del values["time"]
        for m in ("pred", "ground_truth", "proba"):
            if m in metrics_name:
                with open(os.path.join(self.paths["results"], "{}_{}_{}E.txt".format(m, set_name, self.latest_epoch)), "w") as file:
                    file.write("\n".join(values[m]))
                del values[m]
        with open(os.path.join(self.paths["results"], "predict_{}_{}B_{}E.txt".format(set_name, self.params["batch_size"], self.latest_epoch)), "w") as file:
            for key in values.keys():
                file.write("{}: {:.4f}\n".format(key, values[key]))
            file.write("used samples : {}/{}".format(nb_char - self.dataset.nb_ignored[set_name], nb_char))

    def predict_batch(self, params, metrics_name):
        with torch.no_grad():
            x, y, seq_len, seq_reduced_len, labels_len, _, img_name = params
            x = torch.from_numpy(x).float().permute(0, 3, 1, 2).to(self.device)
            y = torch.from_numpy(y).long().to(self.device)

            for model_name in self.models.keys():
                self.models[model_name].eval()

            global_pred = self.models["end_to_end_model"](x)

            truth = [self.ctc_ind_to_str(i) for i in y]

            pred = [self.ctc_decode(pred) for pred in global_pred.permute(0, 2, 1)]

        metrics = {}
        for key in metrics_name:
            if key == "cer":
                metrics[key] = edit_cer_from_list(truth, pred)
            if key == "wer":
                metrics[key] = edit_wer_from_list(truth, pred)
                metrics["nb_words"] = sum([len(t.split(" ")) for t in truth])
            elif key == "pred":
                metrics[key] = pred
            elif key == "ground_truth":
                metrics[key] = truth
            elif key == "diff_len":
                metrics[key] = self.batch_len(truth, pred)
            elif key == "proba":
                metrics[key] = self.batch_probas_to_str(global_pred.cpu().detach().numpy(), img_name)
            elif key == "loss_ctc":
                ctc_loss = CTCLoss(blank=len(self.all_labels))
                metrics[key] = ctc_loss(global_pred.permute(2, 0, 1), y, seq_reduced_len.tolist(),
                                        labels_len.tolist()).item()

        return metrics



