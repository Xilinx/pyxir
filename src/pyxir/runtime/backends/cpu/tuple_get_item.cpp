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
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

#include "pyxir/runtime/runtime.hpp"
#include "tuple_get_item.hpp"
#include "pyxir/opaque_func_registry.hpp"

namespace py = pybind11;

namespace pyxir {
namespace runtime {
namespace cpu {

TupleGetItemFunc::TupleGetItemFunc(XLayerHolder &xl)
  : KernelFunc(xl)
{
  index_ = xl->get_attr("index").get_int();

  transpose_ = xl_->has_attr("transpose") && (xl->get_attr("transpose").get_bool() == true);

  if (transpose_) {
    axes_ = xl_->get_attr("axes").get_ints();

    // Import Python global Transpose function for now
    // auto transpose = py::module::import("pyxir.runtime.globals.transpose");

    if (!pyxir::OpaqueFuncRegistry::Exists("px.globals.Transpose"))
      throw std::runtime_error("Cannot import global Transpose function because"
                              " `px.global.Transpose` opaque function is"
                              " not registered");

    transpose_of_ = pyxir::OpaqueFuncRegistry::Get("px.globals.Transpose");
  }
}

void TupleGetItemFunc::operator()(
  std::vector<XBufferHolder> &in_tensors,
  std::vector<XBufferHolder> &out_tensors)
{
  if (out_tensors.size() == 0) {
    if (transpose_) {
      // Create new output XBuffer
      for (const auto &shape : xl_->shapes) {
        std::vector<ssize_t> buffer_shape = shape;
        buffer_shape[0] = in_tensors[0]->shape[0];
        out_tensors.push_back(create_buffer(buffer_shape));
      }
      
      std::vector<XBufferHolder> transpose_in {in_tensors[index_]};
      // Execute transpose
      transpose_of_(transpose_in, out_tensors, axes_);
    } else {
      out_tensors.push_back(in_tensors[index_]);
    }
  } else {
    // Out tensors are already provided
    assert(out_tensors[0]->size == in_tensors[index_]->size);
    assert(out_tensors[0]->itemsize == in_tensors[index_]->itemsize);

    if (transpose_) {
      std::vector<XBufferHolder> transpose_in {in_tensors[index_]};
      transpose_of_(transpose_in, out_tensors, axes_);
    } else {
      memcpy(out_tensors[0]->data, in_tensors[index_]->data,
             out_tensors[0]->size * out_tensors[0]->itemsize);
    }
  }
    
}

REGISTER_KERNEL_FUNC("cpu.TupleGetItem")
  .set_impl([](XLayerHolder &xl, KernelFuncHolder &kfh) {
    kfh = std::move(KernelFuncHolder(new TupleGetItemFunc(xl)));
  });

} // namespace cpu
} // namespace runtime
} // namespace pyxir
