# Copyright 2020 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Module for classification functions
"""

import numpy as np


def get_predictions(raw_predictions, synset, k, label_lst=None):
    # type: (numpy.ndarray, str, str, int) -> List[dict]
    """
    Retrieve the predicted and correct labels for the given network
    raw_predictions probabilities array
    """
    labels = np.loadtxt(synset, str, delimiter='\n')

    if not len(labels) == raw_predictions.shape[1]:
        raise ValueError("Incompatible labels and predictions shape."
                         " The network makes predictions for {} categories"
                         " but there are {} categories in the labels list"
                         .format(raw_predictions.shape[1], len(labels)))
    if label_lst is not None:
        assert(len(label_lst) == raw_predictions.shape[0])

    predictions = []
    for idx, prediction in enumerate(raw_predictions):
        print("-----------------")

        preds = {
            'predictions': [],
            'correct': None
        }
        top_k = prediction.argsort()[-1:-(k+1):-1]
        for l, p, label_idx in zip(labels[top_k], prediction[top_k], top_k):
            print(l, " : ", p)
            preds['predictions'].append((label_idx, l, p))

        if label_lst:
            correct_label_idx = int(label_lst[idx])
            correct_label = labels[correct_label_idx]
            preds['correct'] = (correct_label_idx, correct_label)
            print("Correct label: {}".format(correct_label))

        predictions.append(preds)

    return predictions


def get_top_k_accuracy(raw_predictions, synset, k, label_lst):
    # type: (numpy.ndarray, str, str, int) -> float
    """
    Return the top_k accuracy for the given raw_predictions with correct labels
    in label_lst
    """

    predictions = get_predictions(raw_predictions, synset, k, label_lst)

    top_k = 0
    for example in predictions:
        for pred_idx, _, _ in example['predictions']:
            if pred_idx == example['correct'][0]:
                top_k += 1

    return float(top_k) / len(label_lst)
