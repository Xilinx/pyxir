/*
 *  Copyright 2020 Xilinx Inc.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *  
 *      http://www.apache.org/licenses/LICENSE-2.0
 *  
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include "pyxir/pyxir_api.hpp"
#include "pyxir/graph/xlayer.hpp"
#include "pyxir/common/xbuffer.hpp"
#include "pyxir/runtime/kernel_func.hpp"

namespace pyxir {
namespace runtime {
namespace cpu {

/**
 * @brief TupleGetItemFunc for executing a TupleGetItem layer, possibly including a transpose
 *  operation.
 */ 
class TupleGetItemFunc : public KernelFunc {

  public:
    TupleGetItemFunc(XLayerHolder &xl);

    void operator()(std::vector<XBufferHolder> &in_tensors,
                    std::vector<XBufferHolder> &out_tensors);

  private:
    // The index indicating the element of the incoming tuple input to be returined
    int index_;
    // Whether to perform a transpose operation after retrieval of the input tensor
    bool transpose_;
    // The transpose axes
    std::vector<int64_t> axes_;
    // Opaque Transpose function for execution
    OpaqueFunc transpose_of_;
};

} // namespace cpu
} // namespace runtime
} // namespace pyxir