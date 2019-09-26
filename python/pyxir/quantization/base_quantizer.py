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
Module for quantizing XGraph models


"""

import abc
import numpy as np


class XGraphBaseQuantizer(object):

    __metaclass__ = abc.ABCMeta

    """

    Attributes
    ----------
    xgraph: XGraph
        the XGraph instance to be quantized
    """

    def __init__(self, xgraph):
        self.xgraph = xgraph

    def _get_calibration_indices(self, nb_inputs, cal_indices, cal_seed,
                                 cal_size):
        # (int, List[int], int, int)-> List[int]
        """
        Return a list of indices that indicate which images should be used for calibration
        """
        # TODO: this basically just gets random indices -> move to util method
        np.random.seed(cal_seed)
        if cal_indices is not None:
            if len(cal_indices) > cal_size:
                raise ValueError('Number of calibration indices exceeds calibration indices size')
            if len(cal_indices) > nb_inputs:
                raise ValueError('Number of calibration indices exceeds number of calibration inputs')
            if max(cal_indices) > nb_inputs:
                raise ValueError('Calibration indices exceed size of calibration inputs')

            remaining_indices = [idx for idx in range(nb_inputs) if not idx in cal_indices]
            extra_cal_indices = np.random.choice(remaining_indices, cal_size - len(cal_indices), replace=False)
            return np.sort(np.append(extra_cal_indices, cal_indices))

        return np.sort(np.random.choice(list(range(nb_inputs)), cal_size, replace=False))

    @abc.abstractmethod
    def quantize(self, inputs=None, stop=None):
        # (numpy.ndarray, str) -> None
        """
        Start quantization of the executable graph model

        Arguments
        ---------
        stop: str (optional, default = None)
            the name of the operation at which to stop quantization
        """
        raise NotImplementedError("")
