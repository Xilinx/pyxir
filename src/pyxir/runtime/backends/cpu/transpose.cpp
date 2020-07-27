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

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

#include "pyxir/runtime/runtime.hpp"
#include "transpose.hpp"

namespace py = pybind11;

namespace pyxir {
namespace runtime {
namespace cpu {

TransposeFunc::TransposeFunc(XLayerHolder &xl)
  : KernelFunc(xl)
{
  axes_ = xl_->get_attr("axes").get_ints();
  // Import Python global Transpose function for now
  // auto transpose = py::module::import("pyxir.runtime.globals.transpose");

  if (!pyxir::OpaqueFuncRegistry::Exists("px.globals.Transpose"))
    throw std::runtime_error("Cannot import global Transpose function because"
                             " `px.global.Transpose` opaque function is"
                             " not registered");

  transpose_of_ = pyxir::OpaqueFuncRegistry::Get("px.globals.Transpose");
}

void TransposeFunc::operator()(
  std::vector<XBufferHolder> &in_tensors,
  std::vector<XBufferHolder> &out_tensors)
{
  if (out_tensors.size() == 0) {
    for (const auto &shape : xl_->shapes) {
      std::vector<ssize_t> buffer_shape = shape;
      buffer_shape[0] = in_tensors[0]->shape[0];
      out_tensors.push_back(create_buffer(buffer_shape));
    }
  }
  transpose_of_(in_tensors, out_tensors, axes_);
}

REGISTER_KERNEL_FUNC("cpu.Transpose")
  .set_impl([](XLayerHolder &xl, KernelFuncHolder &kfh) {
    kfh = std::move(KernelFuncHolder(new TransposeFunc(xl)));
  });

} // namespace cpu
} // namespace runtime
} // namespace pyxir
