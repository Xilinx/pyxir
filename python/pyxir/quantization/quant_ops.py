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
xfDNN module for quantization of neural network graphs in xfDNN intermediate 
representation (IR)


"""

import abc
import numpy as np
import logging

from pyxir.shared.quant_param_factory import LayerParams
from .util import ThresholdLayerInputs, ThresholdLayerOutputs, ThresholdWeights

logger = logging.getLogger("pyxir")


class BaseQuantOp(object):

    __abcmeta__ = abc.ABCMeta

    """
    Attributes
    ----------
    qp_factory: QuantParamFactory
        the quantization parameter factory
    quant_layers: List[tuple]
        the layers to be quantized using the quantization parameter factory
    """

    def __init__(self, qp_factory, quant_layers, bitwidth):
        self._quant_param = qp_factory
        self._quant_layers = quant_layers
        self._bitwidth = bitwidth

    def __call__(self, layer, inpts, outpt, qkey):
        # (RtLayer, List[numpy.ndarray], numpy.ndarray) -> None
        self.quantize(layer, inpts, outpt, qkey)

    @abc.abstractmethod
    def quantize(self, layer, inpts, outpt, qkey):
        # (RtLayer, List[numpy.ndarray], numpy.ndarray, str) -> None
        """
        The quantization method is responsible for quantizing a specific
        layer, given the inputs and output

        TODO: Can we integrate this better with the specified layer instead
        of passing layer objects to this method??
        """
        raise NotImplementedError("")

    '''
    def _find_input_nodes_with_th_set(self, node_name):
        # (str) -> str
        # Some layers (e.g. pooling) are stored under a quant util name
        node_name = node_name + '_QUANT_UTIL' if (node_name + '_QUANT_UTIL')\
            in self._quant_param.th_layer_out else node_name 
        if (node_name in self._quant_param.bw_layer_out and \
            node_name in self._quant_param.th_layer_out):
            return node_name

        for inpt_name in self.runtime.get_input_node_names(node_name):
            d = self._find_input_nodes_with_th_set(inpt_name)
            if d is not None:
                return d

        return None
    '''

    def _internal_name(self, layer_name):
        # type: (str) -> str
        return layer_name + '_QUANT_UTIL' if (layer_name + '_QUANT_UTIL')\
            in self._quant_param.th_layer_out else layer_name 


class SkipQuant(BaseQuantOp):

    def quantize(self, layer, inpts, outpt, qkey):
        # (RtLayer, List[numpy.ndarray], numpy.ndarray, str) -> None
        """
        Skip quantization for the given operation
        """
        pass


class DefaultQuant(BaseQuantOp):

    def quantize(self, layer, inpts, outpt, qkey):
        # (RtLayer, List[numpy.ndarray], numpy.ndarray, str) -> None
        """
        Default quantization for the given operation
        """
        op_name = layer.name

        assert(len(layer.inputs) == len(inpts))
        if len(layer.inputs) != 1:
            # TODO
            logger.warn("[INTERNAL WARNING] DefaultQuant operation executed on"
                        " layer: {} with type: {} has zero or multiple inputs."
                        " Please check if this is correct."
                        .format(layer.name, layer.type))

        if not self._internal_name(op_name) in self._quant_param.th_layer_out:
            # ! check because Relu layer should not adjust its parent layer
            #   quantization params 
            #	(this happens for Relu layer after eltwise layer) -> TODO
            # List is used to make the thresholds point to the same objects
            input_name = self._internal_name(layer.inputs[0])
            th_in_lst = self._quant_param.th_layer_out[input_name]

            self._quant_param.bw_layer_in[op_name] = self._bitwidth
            self._quant_param.th_layer_in[op_name] = th_in_lst
            self._quant_param.bw_layer_out[op_name] = self._bitwidth
            self._quant_param.th_layer_out[op_name] = th_in_lst

# INPUT


class InputQuant(BaseQuantOp):

    def quantize(self, layer, inpts, outpt, qkey):
        # (RtLayer, List[numpy.ndarray], numpy.ndarray, str) -> None
        """
        TODO
        """
        op_name, op_type, params = layer.name, layer.type, layer.get_params()

        logger.info("\nQuantize input layer: {}".format(op_name))

        self._quant_layers[qkey].append((op_name, op_type, None))

        outpt_min, outpt_max, outpt_std = np.min(outpt), np.max(outpt), np.std(outpt)
        
        # TODO Using ThresholdLayerOutputs might return invalid outputs
        #   likely because of invalid Kulback-Leibler divergence parameters
        #   (zeros). This happens for PyTorch (encountered with AlexNet) 
        #   input values.
        #threshold = ThresholdLayerOutputs(outpt, self._bitwidth)
        threshold = ThresholdLayerInputs(outpt, self._bitwidth)

        self._quant_param.bw_layer_in[op_name] = self._bitwidth
        self._quant_param.th_layer_in[op_name] = [threshold] # list to make mutable
        self._quant_param.bw_layer_out[op_name] = self._bitwidth
        self._quant_param.th_layer_out[op_name] = [threshold] # list to make mutable

        logger.debug("Output (n,c,h,w) = {}, Min: {}, Max: {}, Stdev: {}"
                     .format(outpt.shape, outpt_min, outpt_max, outpt_std))

# CONVOLUTION


class ConvQuant(BaseQuantOp):

    def quantize(self, layer, inpts, outpt, qkey):
        # (RtLayer, List[numpy.ndarray], numpy.ndarray, str) -> None
        """
        TODO
        """
        op_name, op_type, params = layer.name, layer.type, layer.get_params()

        logger.info("\nQuantize convolution layer: {}".format(op_name))

        # TODO: adding x to _quant_layers list as layers param, is this correct?
        # TODO: Adding x here -> numpy array not hashable -> add None for now
        self._quant_layers[qkey].append((op_name, op_type, None))


        ## INPUT
        assert(len(inpts) <= 3)
        # TODO: we are now getting the threshold for input layer from a previous layer.
        #   Note that there should exist a previous layer with a threshold as thresholds
        #   are computed on input layers

        # input_names = self.runtime.get_input_node_names(op_name)
        input_name = layer.inputs[0]
        logger.debug("Found input names: {}".format(input_name))
        #assert(len(input_name) == 1)
        #input_node_with_th = self._find_input_nodes_with_th_set(op_name)
        #threshold = self._quant_param.th_layer_out[input_node_with_th]
        int_input_name = self._internal_name(input_name)
        th_in_lst = self._quant_param.th_layer_out[int_input_name]

        self._quant_param.bw_layer_in[op_name] = self._bitwidth
        self._quant_param.th_layer_in[op_name] = th_in_lst

        inpt = inpts[0]
        inpt_min, inpt_max, inpt_std = np.min(inpt), np.max(inpt), np.std(inpt)
        #threshold = ThresholdLayerOutputs(inpt, self._bitwidth) [OLD]
        logger.debug("Input (n,c,h,w) = {}, Min: {}, Max: {}, Stdev: {}".format(inpt.shape, inpt_min, inpt_max, inpt_std))
        

        ## PARAMS
        if len(inpts) > 0 and params['W'] is not None:
            raise ValueError("Convolution kernel should be passed either as an input"\
                " or a parameter but not both")
        elif len(inpts) > 1:
            weights = inpts[1]
        else:
            weights = params["W"]
            weights = np.transpose(weights, (3,2,0,1))
        weights_min, weights_max, weights_std = np.min(weights), np.max(weights), np.std(weights)
        
        # weights should have format OIHW, which they should already be??
        # TODO: pass data_layout??
        threshold = ThresholdWeights(weights, self._bitwidth)

        self._quant_param.bw_params[op_name] = self._bitwidth
        self._quant_param.th_params[op_name] = threshold

        logger.debug("Weights (outchan,inchan,h,w) = {}, Min: {}, Max: {}, Stdev: {}".format(weights.shape, weights_min, weights_max, weights_std))


        ## OUTPUTS
        outpt_min, outpt_max, outpt_std = np.min(outpt), np.max(outpt), np.std(outpt)
        threshold = ThresholdLayerOutputs(outpt, self._bitwidth)
        logger.debug(threshold)

        self._quant_param.bw_layer_out[op_name] = self._bitwidth
        self._quant_param.th_layer_out[op_name] = [threshold]
        
        logger.debug("Output (n,c,h,w) = {}, Min: {}, Max: {}, Stdev: {}".format(outpt.shape, outpt_min, outpt_max, outpt_std))

## SCALE

class ScaleQuant(BaseQuantOp):

    def quantize(self, layer, inpts, outpt, qkey):
        # (RtLayer, List[numpy.ndarray], numpy.ndarray, str) -> None
        """
        TODO
        """
        op_name, op_type, params = layer.name, layer.type, layer.get_params()

        logger.info("\nQuantize scaling layer: {}".format(op_name))

        ## INPUT

        # input_names = self.runtime.get_input_node_names(op_name)
        # logger.debug("Found input names: {}".format(input_names))
        # assert(len(input_names) == 1)
        # input_node_with_th = self._find_input_nodes_with_th_set(op_name)
        # threshold = self._quant_param.th_layer_out[input_node_with_th]

        input_name = layer.inputs[0]
        logger.debug("Found input names: {}".format(input_name))
        int_input_name = self._internal_name(input_name)
        th_in_lst = self._quant_param.th_layer_out[int_input_name]

        self._quant_param.bw_layer_in[op_name] = self._bitwidth
        self._quant_param.th_layer_in[op_name] = th_in_lst

        inpt = inpts[0]
        inpt_min, inpt_max, inpt_std = np.min(inpt), np.max(inpt), np.std(inpt)
        #threshold = ThresholdLayerOutputs(inpt, self._bitwidth) [OLD]
        logger.debug("Input (n,c,h,w) = {}, Min: {}, Max: {}, Stdev: {}".format(inpt.shape, inpt_min, inpt_max, inpt_std))
        
        ## PARAMS

        if len(inpts) > 0 and params['gamma'] is not None:
            raise ValueError("scaling parameter should be passed either as an "
                "input or a parameter but not both")
        elif len(inpts) > 1:
            gamma = inpts[1]
        else:
            gamma = params["gamma"]
            #weights = np.transpose(weights, (3,2,0,1))

        # weights should have format IOHW, which they should already be??
        # TODO: pass data_layout??
        th_params = ThresholdWeights(gamma, self._bitwidth)

        #logger.debug("th_params: {}".format(th_params))

        self._quant_param.bw_params[op_name] = self._bitwidth
        self._quant_param.th_params[op_name] = th_params

        gamma_min, gamma_max, gamma_std = \
            np.min(gamma), np.max(gamma), np.std(gamma)

        logger.debug("Weights (channels) = {}, Min: {}, Max: {}, Stdev: {}"
            .format(gamma.shape, gamma_min, gamma_max, gamma_std))

        ## OUTPUT

        # Only calculate threshold on outputs
        outpt_min, outpt_max, outpt_std = np.min(outpt), np.max(outpt), np.std(outpt)
        threshold = ThresholdLayerOutputs(outpt, self._bitwidth)

        self._quant_param.bw_layer_out[op_name] = self._bitwidth
        self._quant_param.th_layer_out[op_name] = [threshold]

        # TODO: do Float2Fixed2Float??

        logger.debug("Output (n,c,h,w) = {}, Min: {}, Max: {}, Stdev: {}"
            .format(outpt.shape, outpt_min, outpt_max, outpt_std))

        # sf_params = th_params / (np.power(2.0, self._bitwidth - 1) - 1)
        # TODO: Adding th_params / 2**(bitwidth-1) -1 = sf_params as layer params
        # to make sure that the 'clip' part in calculating the multiplier in 
        # quantize_base.py becomes equal to 1
        # TODO: why is this clip part there?????

        # !! scaling is done by an elementwise addition folowwed by scale and shift 
        #   from quantization parameters, instead of an explicit scale
        self._quant_layers[qkey].append(
            [op_name, op_type, [LayerParams(gamma)]])

## BATCHNORM

class BatchNormQuant(BaseQuantOp):

    def quantize(self, layer, inpts, outpt, qkey):
        # (RtLayer, List[numpy.ndarray], numpy.ndarray, str) -> None
        """
        TODO
        """
        raise NotImplementedError("")

## CONCAT

class ConcatQuant(BaseQuantOp):

    def quantize(self, layer, inpts, outpt, qkey):
        # (RtLayer, List[numpy.ndarray], numpy.ndarray, str) -> None
        """
        TODO
        """
        op_name, op_type, params = layer.name, layer.type, layer.get_params()

        # TODO: Overlap with quantize_tf.py

        logger.info("Quantize concat layer: {}".format(op_name))
        self._quant_layers[qkey].append((op_name, op_type, None))

        # Only calculate threshold on outputs
        outpt_min, outpt_max, outpt_std = np.min(outpt), np.max(outpt), np.std(outpt)
        threshold = ThresholdLayerOutputs(outpt, self._bitwidth)

        for input_name in layer.inputs:
            int_input_name = self._internal_name(input_name)
            self._quant_param.th_layer_out[int_input_name][0] = threshold

        input_name = self._internal_name(layer.inputs[0])
        th_lst = self._quant_param.th_layer_out[input_name]

        self._quant_param.th_layer_in[op_name] = th_lst 
        self._quant_param.th_layer_out[op_name] = th_lst[:] # TODO for DenseNet-like architectures
        self._quant_param.bw_layer_in[op_name] = self._bitwidth
        self._quant_param.bw_layer_out[op_name] = self._bitwidth

        # for inpt_name in self.runtime.get_input_node_names(op_name):
        #     d = self._find_input_nodes_with_th_set(inpt_name)
        #     logger.debug("Inpt name: {}, find: {}".format(inpt_name, d))
        #     if d is not None:
        #         self._quant_param.bw_layer_out[d] = self._bitwidth
        #         self._quant_param.th_layer_out[d] = threshold

        # TODO How to handle quantization for concat layers after concat layers?
        #   See DenseNet kind of architectures.

        input_names = layer.inputs
        for input_name in input_names:
            input_name = self._internal_name(input_name)
            self._quant_param.th_layer_out[input_name][0] = th_lst[0]

        logger.debug("Output (n,c,h,w) = {}, Min: {}, Max: {}, Stdev: {}"
            .format(outpt.shape, outpt_min, outpt_max, outpt_std))

class ConcatQuantWithScale(BaseQuantOp):

    def quantize(self, layer, inpts, outpt, qkey):
        # (RtLayer, List[numpy.ndarray], numpy.ndarray, str) -> None
        """
        TODO
        """
        op_name, op_type, params = layer.name, layer.type, layer.get_params()

        # TODO: Overlap with quantize_tf.py

        logger.info("Quantize concat layer: {}".format(op_name))
        self._quant_layers[qkey].append((op_name, op_type, None))

        # Only calculate threshold on outputs
        outpt_min, outpt_max, outpt_std = np.min(outpt), np.max(outpt), np.std(outpt)
        threshold = ThresholdLayerOutputs(outpt, self._bitwidth)

        self._quant_param.th_layer_in[op_name] = [threshold]
        self._quant_param.th_layer_out[op_name] = [threshold] # TODO for DenseNet-like architectures
        self._quant_param.bw_layer_in[op_name] = self._bitwidth
        self._quant_param.bw_layer_out[op_name] = self._bitwidth

        # TODO How to handle quantization for concat layers after concat layers?
        #   See DenseNet kind of architectures.

        logger.debug("Output (n,c,h,w) = {}, Min: {}, Max: {}, Stdev: {}"
            .format(outpt.shape, outpt_min, outpt_max, outpt_std))


## ELTWISE

class EltwiseQuant(BaseQuantOp):

    def quantize(self, layer, inpts, outpt, qkey):
        # (RtLayer, List[numpy.ndarray], numpy.ndarray, str) -> None
        """
        TODO
        """
        op_name, op_type, params = layer.name, layer.type, layer.get_params()

        logger.info("Quantize elemtwise layer: {}".format(op_name))

        self._quant_layers[qkey].append((op_name, op_type, None))
        
        ## INPUT

        # TODO: First add inputs and then compute new threshold and set thresholds of 
        #   inputs to this new threshold??
        inpt_min, inpt_max, inpt_std = np.min(np.array(inpts))\
            , np.max(np.array(inpts)), np.std(np.array(inpts))
        # # Get thresholds from inputs th_out
        # input_names = self.runtime.get_input_node_names(op_name)
        # logger.debug("Found input names: {}".format(input_names))
        # assert(len(input_names) == 2)

        # input_nodes_with_th = \
        #     [self._find_input_nodes_with_th_set(in_name) for in_name in input_names]
        # threshold = self._quant_param.th_layer_out[max(input_nodes_with_th, 
        #     key=lambda in_name: self._quant_param.th_layer_out[in_name])]

        input_names = layer.inputs
        th_in_lst = self._quant_param.th_layer_out[
            self._internal_name(max(input_names, key=lambda in_name: 
                self._quant_param.th_layer_out[self._internal_name(in_name)]))]

        # threshold = ThresholdLayerOutputs(inpts[0], self._bitwidth)
        # for i in range(1, len(inpts)):
        #     threshold = max(threshold, ThresholdLayerOutputs(inpts[i], self._bitwidth)) 
        #threshold = np.maximum(threshold_left, threshold_right)
        logger.debug("Threshold in: {}".format(th_in_lst))

        # Set the output thresholds of the inputs with thresholds set to make sure
        #   that all inputs are on the same scale before being added
        # TODO
        # for inpt_name in self.runtime.get_input_node_names(op_name):
        #     d = self._find_input_nodes_with_th_set(inpt_name)
        #     logger.debug("Set threshold for ancestor: {}".format(d))
        #     if d is None:
        #         raise ValueError("No input ancestor to elementwise operation: {} "\
        #             "with threshold set. This is required.")
        #     self._quant_param.bw_layer_out[d] = self._bitwidth
        #     self._quant_param.th_layer_out[d] = threshold

        for input_name in input_names:
            input_name = self._internal_name(input_name)
            self._quant_param.th_layer_out[input_name][0] = th_in_lst[0]

        self._quant_param.bw_layer_in[op_name] = self._bitwidth
        self._quant_param.th_layer_in[op_name] = th_in_lst

        logger.debug("Input left shape: {}, right shape:{},  Min: {}, Max: {}, "\
            "Stdev: {}".format(inpts[0].shape, inpts[1].shape, inpt_min, 
                                inpt_max, inpt_std))

        # Only calculate threshold on outputs
        outpt_min, outpt_max, outpt_std = np.min(outpt), np.max(outpt), np.std(outpt)
        threshold_out = ThresholdLayerOutputs(outpt, self._bitwidth)

        logger.debug("Threshold out: {}".format(threshold_out))

        self._quant_param.bw_layer_out[op_name] = self._bitwidth
        self._quant_param.th_layer_out[op_name] = [threshold_out]

        logger.debug("Output (n,c,h,w) = {}, Min: {}, Max: {}, Stdev: {}"
            .format(outpt.shape, outpt_min, outpt_max, outpt_std))


class EltwiseQuantWithScale(BaseQuantOp):

    def quantize(self, layer, inpts, outpt, qkey):
        # (RtLayer, List[numpy.ndarray], numpy.ndarray, str) -> None
        """
        TODO
        """
        op_name, op_type, params = layer.name, layer.type, layer.get_params()

        logger.info("Quantize elemtwise layer: {}".format(op_name))

        self._quant_layers[qkey].append((op_name, op_type, None))
        
        ## INPUT

        # TODO: First add inputs and then compute new threshold and set thresholds of 
        #   inputs to this new threshold??
        inpt_min, inpt_max, inpt_std = np.min(np.array(inpts))\
            , np.max(np.array(inpts)), np.std(np.array(inpts))
        # Get thresholds from inputs th_out
        # input_names = self.runtime.get_input_node_names(op_name)
        # logger.debug("Found input names: {}".format(input_names))
        # assert(len(input_names) == 2)

        # input_nodes_with_th = \
        #     [self._find_input_nodes_with_th_set(in_name) for in_name in input_names]
        # threshold = self._quant_param.th_layer_out[max(input_nodes_with_th, 
        #     key=lambda in_name: self._quant_param.th_layer_out[in_name])]
        input_names = layer.inputs
        th_in_lst = self._quant_param.th_layer_out[
            self._internal_name(max(input_names, key=lambda in_name: 
                self._quant_param.th_layer_out[self._internal_name(in_name)]))]

        # threshold = ThresholdLayerOutputs(inpts[0], self._bitwidth)
        # for i in range(1, len(inpts)):
        #     threshold = max(threshold, ThresholdLayerOutputs(inpts[i], self._bitwidth)) 
        #threshold = np.maximum(threshold_left, threshold_right)
        logger.debug("Threshold in: {}".format(th_in_lst))

        # Set the output thresholds of the inputs with thresholds set to make sure
        #   that all inputs are on the same scale before being added
        # TODO
        # for inpt_name in self.runtime.get_input_node_names(op_name):
        #     d = self._find_input_nodes_with_th_set(inpt_name)
        #     logger.debug("Set threshold for ancestor: {}".format(d))
        #     if d is None:
        #         raise ValueError("No input ancestor to elementwise operation: {} "\
        #             "with threshold set. This is required.")
        #     self._quant_param.bw_layer_out[d] = self._bitwidth
        #     self._quant_param.th_layer_out[d] = threshold

        self._quant_param.bw_layer_in[op_name] = self._bitwidth
        self._quant_param.th_layer_in[op_name] = th_in_lst

        logger.debug("Input left shape: {}, right shape:{},  Min: {}, Max: {}, "\
            "Stdev: {}".format(inpts[0].shape, inpts[1].shape, inpt_min, 
                                inpt_max, inpt_std))

        # Only calculate threshold on outputs
        outpt_min, outpt_max, outpt_std = np.min(outpt), np.max(outpt), np.std(outpt)
        threshold_out = ThresholdLayerOutputs(outpt, self._bitwidth)

        logger.debug("Threshold out: {}".format(threshold_out))

        self._quant_param.bw_layer_out[op_name] = self._bitwidth
        self._quant_param.th_layer_out[op_name] = [threshold_out]

        logger.debug("Output (n,c,h,w) = {}, Min: {}, Max: {}, Stdev: {}"
            .format(outpt.shape, outpt_min, outpt_max, outpt_std))


##POOLING

class PoolQuant(BaseQuantOp):

    def quantize(self, layer, inpts, outpt, qkey):
        # (RtLayer, List[numpy.ndarray], numpy.ndarray, str) -> None
        """
        TODO
        """
        if layer.op == 'Max':
            return self._quantize_max(layer, inpts, outpt, qkey)
        elif layer.op =='Avg':
            return self._quantize_avg(layer, inpts, outpt, qkey)
        else:
            raise ValueError("Unsupported pooling operation: {}. Only `Max` "\
                " and `Avg` are valid operations.")
    
    def _quantize_max(self, layer, inpts, outpt, qkey):
        # (RtLayer, List[numpy.ndarray], numpy.ndarray, str) -> None
        """
        TODO Describe pool quantization in high level here
        """
        assert(layer.op == 'Max')
        
        op_name, op_type, params = layer.name, layer.type, layer.get_params()

        logger.info("Quantize max pooling layer: {}".format(op_name))

        # TODO Change name of max pool because FPGA does wrong things if 
        #   the corresponding quantization parameters are provided
        quant_name = op_name + '_QUANT_UTIL'

        # NOTE: Avg pool computes sum instead of average on FPGA, division is left to
        #   the quantization parameters so the pool_divisor should be included 
        #   for average pooling. For maxpool this divisor is not necessary and
        #   we set it to 1
        pool_divisor = [1]
        self._quant_layers[qkey].append(
            (quant_name, op_type, LayerParams(pool_divisor)))


        ## INPUT
        inpt = inpts[0]
        inpt_min, inpt_max, inpt_std = np.min(inpt), np.max(inpt), np.std(inpt)
        # th_in = ThresholdLayerOutputs(inpt, self._bitwidth)
        # input_node_with_th = self._find_input_nodes_with_th_set(op_name)
        assert(len(layer.inputs) == 1)
        input_name = self._internal_name(layer.inputs[0])
        th_in_lst = self._quant_param.th_layer_out[input_name]

        self._quant_param.bw_layer_in[quant_name] = self._bitwidth
        self._quant_param.th_layer_in[quant_name] = th_in_lst

        logger.debug("Input (n,c,h,w) = ({}), Min: {}, Max: {}, Stdev: {}"
            .format(inpt.shape, inpt_min, inpt_max, inpt_std))
        
        ## NO PARAMS

        ## OUTPUTS
        outpt_min, outpt_max, outpt_std = np.min(outpt), np.max(outpt), np.std(outpt)
        # th_out = ThresholdLayerOutputs(outpt, self._bitwidth)

        self._quant_param.bw_layer_out[quant_name] = self._bitwidth
        self._quant_param.th_layer_out[quant_name] = th_in_lst
        
        logger.debug("Output (n,c,h,w) = ({}), Min: {}, Max: {}, Stdev: {}"
            .format(outpt.shape, outpt_min, outpt_max, outpt_std))
    
    def _quantize_avg(self, layer, inpts, outpt, qkey):
        # (RtLayer, List[numpy.ndarray], numpy.ndarray) -> None
        """
        TODO Describe pool quantization in high level here
        """
        assert(layer.op == 'Avg')
        
        op_name, op_type, params = layer.name, layer.type, layer.get_params()
        
        logger.info("Quantize average pool layer: {}".format(op_name))

        logger.debug("Kernel product = {}".format(np.prod(layer.ksize)))
        quant_name = op_name
        # NOTE: Avg pool computes sum instead of average on FPGA, division is left to
        #   the quantization parameters so the pool_divisor should be included 
        #   here for average pooling
        pool_divisor = [np.prod(layer.ksize)]
        self._quant_layers[qkey].append(
            (op_name, op_type, LayerParams(pool_divisor)))


        ## INPUT
        inpt = inpts[0]
        inpt_min, inpt_max, inpt_std = np.min(inpt), np.max(inpt), np.std(inpt)
        
        # th_in = ThresholdLayerOutputs(inpt, self._bitwidth)
        # input_node_with_th = self._find_input_nodes_with_th_set(op_name)
        # th_in = self._quant_param.th_layer_out[input_node_with_th]
        assert(len(layer.inputs) == 1)
        input_name = self._internal_name(layer.inputs[0])
        th_in_lst = self._quant_param.th_layer_out[input_name]

        self._quant_param.bw_layer_in[quant_name] = self._bitwidth
        self._quant_param.th_layer_in[quant_name] = th_in_lst

        logger.debug("Input (n,c,h,w) = ({}), Min: {}, Max: {}, Stdev: {}"
            .format(inpt.shape, inpt_min, inpt_max, inpt_std))
        
        ## NO PARAMS

        ## OUTPUTS
        outpt_min, outpt_max, outpt_std = np.min(outpt), np.max(outpt), np.std(outpt)
        th_out = ThresholdLayerOutputs(outpt, self._bitwidth)

        self._quant_param.bw_layer_out[quant_name] = self._bitwidth
        self._quant_param.th_layer_out[quant_name] = [th_out]
        
        logger.debug("Output (n,c,h,w) = ({}), Min: {}, Max: {}, Stdev: {}"
            .format(outpt.shape, outpt_min, outpt_max, outpt_std))
