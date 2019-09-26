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

""" Base optimizer module """

import abc
import numpy as np


class XfBaseOptimizer(object):

    __metaclass__ = abc.ABCMeta

    """
    Responsible for optimizing an executable runtime graphs

    Arguments:
    ----------
    runtime: BaseRuntime
        The executable Runtime to be optimized

    Attributes:
    -----------

    """

    def __init__(self, runtime):
        # type: (BaseRuntime) -> XfBaseOptimizer
        self.runtime = runtime

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle):
        # type: () -> numpy.ndarray, numpy.ndarray
        """
        TODO
        """
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    @abc.abstractmethod
    def optimize(self,
                 input_name,
                 X_train,
                 y_train,
                 batch_size,
                 num_epochs=1,
                 X_valid=None,
                 y_valid=None):
        # type: (str, numpy.ndarray, numpy.ndarray, int, int,numpy.ndarray
        #   numpy.ndarray) -> dict
        """
        Optimize the provided runtime

        TODO: Multiple inputs
        Arguments
        ---------
        input_name: str
            the input name of the provided runtime
        X_train: numpy.ndarray
            the data to be used for optimization
        y_train: numpy.ndarray
            the golden labels to be used for optimization
        batch_size: int
            the optimization batch size
        num_epochs: int
            the number of epochs
        X_valid: numpy.ndarray
            the data to be used for validation
        y_valid: numpy.ndarray
            the labels to be used for validation
        """
        raise NotImplementedError("")

    