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

#include <cassert>

#include "pyxir/runtime/runtime.hpp"
#include "input.hpp"

namespace pyxir {
namespace runtime {
namespace cpu {

InputFunc::InputFunc(XLayerHolder &xl)
  : KernelFunc(xl)
{}

void InputFunc::operator()(
  std::vector<XBufferHolder> &in_tensors,
  std::vector<XBufferHolder> &out_tensors)
{
  assert(in_tensors.size() == 1);
  if (out_tensors.size() == 0)
    out_tensors.push_back(in_tensors[0]);
  else
    out_tensors[0] = in_tensors[0];
}

REGISTER_KERNEL_FUNC("cpu.Input")
  .set_impl([](XLayerHolder &xl, KernelFuncHolder &kfh) {
    kfh = std::move(KernelFuncHolder(new InputFunc(xl)));
  });

} // namespace cpu
} // namespace runtime
} // namespace pyxir
