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

#include <cstdlib>

#include "pyxir/runtime/compute_func_factory.hpp"
#include "online_quant_compute_func.hpp"

namespace pyxir {
namespace runtime {

OnlineQuantComputeFunc::OnlineQuantComputeFunc(
  XGraphHolder &xg,
  const std::string &target,
  const std::vector<std::string> &in_tensor_names,
  const std::vector<std::string> &out_tensor_names,
  const std::string &runtime,
  int nb_quant_inputs,
  bool compile_only)
  : xg_(xg), target_(target), in_tensor_names_(in_tensor_names),
    out_tensor_names_(out_tensor_names), runtime_(runtime),
    nb_quant_inputs_(nb_quant_inputs), compile_only_(compile_only)
{
  OpaqueFunc build_online_quant_rt_func =
    OpaqueFuncRegistry::Get("pyxir.build_online_quant_rt");

  // If PX_BUILD_DIR environment variable is set, we skip on-the-fly quantization
  //  and create the final ComputeFunc instead
  const char *env_tmp = std::getenv("PX_BUILD_DIR");
  if (env_tmp != NULL) {
    cf_ = ComputeFuncFactory::GetComputeFunc(
      xg_, target_, in_tensor_names_, out_tensor_names_, runtime_
    ); 

    // Make sure quantization function doesn't get called
    count_ = nb_quant_inputs_ + 1;
  } else {   
    quant_of_ = std::make_shared<OpaqueFunc>(OpaqueFunc());
    OpaqueFuncHolder rt_func_of = std::make_shared<OpaqueFunc>(OpaqueFunc());
    build_online_quant_rt_func(xg_, target_, runtime_, in_tensor_names_,
      out_tensor_names_, quant_of_, rt_func_of);
	  cf_ = ComputeFuncHolder(new OpaqueComputeFunc(rt_func_of));
  }
}

void OnlineQuantComputeFunc::operator()(
  std::vector<XBufferHolder> &in_tensors,
  std::vector<XBufferHolder> &out_tensors)
{
  (*cf_)(in_tensors, out_tensors);
  ++count_;
  
  if (count_ == nb_quant_inputs_) {
    // Call quantization function
    OpaqueArgs args = OpaqueArgs();
    quant_of_->call(args);

    if (compile_only_) {
      if (!OpaqueFuncRegistry::Exists("pyxir.compile"))
        throw std::runtime_error("Cannot compile the Vitis-AI compute func because the "
                                " `pyxir.compile` opaque function is not "
                                " registered. Check if Pyxir python module"
                                " is imported correctly.");

      XGraphHolder scheduled_xg =
        std::make_shared<pyxir::graph::XGraph>("scheduled_xgraph"); 

      OpaqueFunc compile_func = OpaqueFuncRegistry::Get("pyxir.compile");
      
      compile_func(xg_, target_, in_tensor_names_, out_tensor_names_, scheduled_xg);

      pxWarning("Not switching to specified runtime: `" + runtime_ + "` after on-the-fly" +
                " quantization as the model is compiled for a different target device.");
    } else {
      cf_ = ComputeFuncFactory::GetComputeFunc(
        xg_, target_, in_tensor_names_, out_tensor_names_, runtime_
      );
    }
  }
}

} // namespace runtime
} // namespace pyxir
