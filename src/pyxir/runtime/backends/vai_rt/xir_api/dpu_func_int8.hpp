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

#include <unordered_set>

#include "pyxir/pyxir_api.hpp"
#include "pyxir/graph/xlayer.hpp"
#include "pyxir/common/xbuffer.hpp"
#include "pyxir/runtime/kernel_func.hpp"
#include "common.hpp"
#include "dpu_func.hpp"


namespace pyxir {
namespace runtime {
namespace vai_rt {
	
class DpuFuncInt8 : public DpuFunc {

public:
    DpuFuncInt8(XLayerHolder &xl, const std::string &build_dir) : DpuFunc(xl, build_dir){}
    void operator()(std::vector<XBufferHolder> &in_tensors,
                    std::vector<XBufferHolder> &out_tensors);

};

} // vai_rt
} // namespace runtime
} // namespace pyxir

