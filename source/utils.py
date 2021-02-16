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

import numpy as np
import torch
import editdistance


def pad_sequences_1d(data, padding_value=0.0):
    x_lengths = [len(x) for x in data]
    longest_x = max(x_lengths)
    padded_data = np.ones((len(data), longest_x)) * padding_value
    for i, x_len in enumerate(x_lengths):
        padded_data[i, :x_len] = data[i][:x_len]
    return padded_data


def pad_sequences_2d(data, padding_value=0.0):
    x_lengths = [x.shape[0] for x in data]
    y_lengths = [x.shape[1] for x in data]
    longest_x = max(x_lengths)
    longest_y = max(y_lengths)
    if len(data[0].shape) == 3:
        padded_data = np.ones((len(data), longest_x, longest_y, data[0].shape[2])) * padding_value
    else:
        padded_data = np.ones((len(data), longest_x, longest_y)) * padding_value
    for i, xy_len in enumerate(zip(x_lengths, y_lengths)):
        x_len, y_len = xy_len
        padded_data[i, :x_len, :y_len, ...] = data[i][:x_len, :y_len, ...]
    return padded_data


def edit_cer_from_list(truth, pred):
    edit = 0
    for t, p in zip(truth, pred):
        edit += editdistance.eval(t, p)
    return edit


def edit_wer_from_list(truth, pred):
    edit = 0
    for t, p in zip(truth, pred):
        edit += editdistance.eval(t.split(" "), p.split(" "))
    return edit


def cer_from_list_str(str_gt, str_pred):
    len_ = 0
    edit = 0
    for pred, gt in zip(str_pred, str_gt):
        gt = gt.strip("\n")
        pred = pred.strip("\n")
        edit += editdistance.eval(gt, pred)
        len_ += len(gt)
    return edit / len_


def wer_from_list_str(str_gt, str_pred):
    separation_marks = ["?", ".", ";", ",", "!"]
    len_ = 0
    edit = 0
    for pred, gt in zip(str_pred, str_gt):
        gt = gt.strip("\n")
        pred = pred.strip("\n")
        for mark in separation_marks:
            gt.replace(mark, " {} ".format(mark))
            pred.replace(mark, " {} ".format(mark))
        gt = gt.split(" ")
        pred = pred.split(" ")
        while '' in gt:
            gt.remove('')
        while '' in pred:
            pred.remove('')
        edit += editdistance.eval(gt, pred)
        len_ += len(gt)
    cer = edit / len_
    return cer


def cer_from_files(file_gt, file_pred):
    with open(file_pred, "r") as f_p:
        str_pred = f_p.readlines()
    with open(file_gt, "r") as f_gt:
        str_gt = f_gt.readlines()
    return cer_from_list_str(str_gt, str_pred)


def wer_from_files(file_gt, file_pred):
    with open(file_pred, "r") as f_p:
        str_pred = f_p.readlines()
    with open(file_gt, "r") as f_gt:
        str_gt = f_gt.readlines()
    return wer_from_list_str(str_gt, str_pred)


def model_summary(model, shape, device):

    def compute_nb_params(module):
        return sum([np.prod(p.size()) for p in list(module.parameters())])

    layers_info = list()
    layers_info.append({
        "name": "Input",
        "shape": np.array(shape[1:]),
        "details": "",
        "params": 0
    })

    def add_layer(m, in_, res):

        name = str(m.__class__).split(".")[-1].split("'")[0]
        previous_shape = np.array((layers_info[-1]["shape"]))
        if name == "Sequential":
            pass
        elif name in ("Conv2d", "MaxPool2d"):
            padding = np.array(m.padding)
            stride = np.array(m.stride)
            kernel = np.array(m.kernel_size)
            dilation = np.array(m.dilation)
            if name == "Conv2d":
                previous_shape[0] = m.out_channels
            previous_shape[1:] = \
                np.array(((previous_shape[1:] + 2 * padding - (dilation*(kernel-1)+1)) / stride + 1).astype(np.int32))
            layers_info.append({
                "name": name,
                "shape": previous_shape,
                "details": "K{} P{}, S{} ; D{}".format(kernel, padding, stride, dilation),
                "params": compute_nb_params(m),
                "kernel": kernel,
                "padding": padding,
                "dilation": dilation,
                "stride": stride,
                "out_channels": previous_shape[0],
            })
        elif name == "Conv2dPaddingSame":
            stride = layers_info[-1]["stride"]
            kernel = layers_info[-1]["kernel"]
            dilation = layers_info[-1]["dilation"]
            previous_shape = np.array((layers_info[-2]["shape"]))
            previous_shape[0] = layers_info[-1]["out_channels"]
            layers_info[-1] = {
                "name": name,
                "shape": previous_shape,
                "details": "K{} P{}, S{} ; D{}".format(kernel, "same", stride, dilation),
                "params": compute_nb_params(m)
            }
        elif name == "ZeroPad2d":
            layers_info[-1] = {
                "name": name,
                "shape": previous_shape + [0, m.padding[2]+m.padding[3], m.padding[0]+m.padding[1]],
                "details": "P{}".format(m.padding),
                "params": 0
            }
        elif name == "LSTM":
            previous_shape[0] = m.hidden_size
            if m.bidirectional:
                previous_shape[0] *= 2
            layers_info.append({
                "name": name,
                "shape": previous_shape,
                "details": "HS:{} ; IS:{} ; Num:{} ; blstm:{} ; D:{}".format(m.hidden_size, m.input_size, m.num_layers, m.bidirectional, m.dropout),
                "params": compute_nb_params(m)
            })
        elif name == "Linear":
            previous_shape[0] = m.out_features
            layers_info.append({
                "name": name,
                "shape": previous_shape,
                "details": "",
                "params": compute_nb_params(m)
            })

    for m in list(model.modules()):
        m.register_forward_hook(add_layer)

    x = torch.ones(shape).to(device)
    x = model(x)

    print("\n\n{:^20s} - {:^50s} - {:^15s} - {:^10s}".format("Name", "Details", "Shape", "Params"))
    for layer in layers_info:
        print("{:^20s} - {:^50s} - {:^15s} - {:^10s}".format(layer["name"], layer["details"],
                                                             str(tuple(layer["shape"])), str(layer["params"])))
    print("\nTotal params : {}".format("{:,}".format(compute_nb_params(model))))


def receptive_field(model, shape, device):
    layers_info = list()
    layers_info.append({
        "name": "input",
        "shape": np.array(shape[2:]),
        "rf": np.array([1, 1]),
        "j": np.array([1, 1]),
        "start": np.array([0.5, 0.5])
    })

    def layer_receptive_field(module, in_, res_):
        class_name = str(module.__class__).split(".")[-1].split("'")[0]
        if class_name in ("Conv2d", "MaxPool2d", "AdaptiveMaxPooling2d"):
            add_conv_layer(module, class_name)
        elif class_name == "ZeroPad2d":
            add_padding_layer(module, class_name)

    def add_padding_layer(layer, class_name):
        previous_layer = layers_info[-1]
        pad_w = layer.padding[0] + layer.padding[1]
        pad_h = layer.padding[2] + layer.padding[3]
        pad = np.array([pad_h, pad_w])
        new_layer = {
            "name": class_name,
            "shape": np.array(previous_layer["shape"] + pad).astype(np.int32),
            "j": previous_layer["j"],
            "rf": previous_layer["rf"],
            "start": previous_layer["start"],
        }
        layers_info.append(new_layer)

    def add_conv_layer(layer, class_name):
        padding = np.array(layer.padding)
        stride = np.array(layer.stride)
        kernel = np.array(layer.kernel_size)
        # dilation = np.array(layer.dilation)
        previous_layer = layers_info[-1]
        new_layer = {
            "name": class_name,
            "shape": np.array((previous_layer["shape"] + 2 * padding - kernel) / stride + 1).astype(np.int32),
            "j": previous_layer["j"] * stride,
            "rf": previous_layer["rf"] + (kernel - 1) * previous_layer["j"],
            "start": previous_layer["start"] + ((kernel - 1) / 2 - padding) * previous_layer["j"],
            "padding": padding,
            "stride": stride,
            "kernel": kernel,
        }
        layers_info.append(new_layer)

    for m in list(model.modules()):
        m.register_forward_hook(layer_receptive_field)
    x = torch.ones(shape).to(device)
    _ = model(x)

    print("\n\n{:^20s} - {:^15s} - {:^15s}".format("Layer name", "Shape", "RF"))
    for layer in layers_info:
        print("{:^20s} - {:^15s} - {:^15s}".format(layer["name"], str(tuple(layer["shape"])),
                                                   str(tuple(layer["rf"]))))
