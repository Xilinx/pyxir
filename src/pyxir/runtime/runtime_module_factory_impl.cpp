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
#include <iostream>
#include <memory>


#include "pyxir/opaque_func_registry.hpp"
#include "pyxir/runtime/compute_func_factory.hpp"
#include "pyxir/runtime/runtime_module_factory_impl.hpp"
#include "pyxir/runtime/online_quant_compute_func.hpp"


namespace pyxir {
namespace runtime {

RtModHolder DefaultRuntimeModuleFactoryImpl::get_runtime_module(
    std::shared_ptr<graph::XGraph> &xg, const std::string &target,
    const std::vector<std::string> &in_tensor_names,
    const std::vector<std::string> &out_tensor_names,
    RunOptionsHolder run_options) {
  if (!OpaqueFuncRegistry::Exists("pyxir.build_rt"))
    throw std::runtime_error("Cannot build the runtime because the "
                             " `pyxir.build_rt` opaque function is not "
                             " registered. Check if Pyxir python module"
                             " is imported correctly.");

  // If the on-the-fly quantization option is enabled, we create a compute function
  //  that performs quantization calibration on the first N inputs (and computes
  //  actual results on CPU) and afterwards switches to hardware acceleration
  // NOTE: If PX_BUILD_DIR environment variable is set, we skip on-the-fly
  //  quantization and create the ComputeFunc immediately
  const char *env_px_build = std::getenv("PX_BUILD_DIR");
  if (run_options && run_options->on_the_fly_quantization
      && run_options->nb_quant_inputs > 0 && env_px_build == NULL)
  {
    if (!OpaqueFuncRegistry::Exists("pyxir.quantize"))
      throw std::runtime_error("Cannot runtime with on-the-fly quantization"
                               " because the "
                               " `pyxir.quantize` opaque function is not "
                               " registered. Check if Pyxir python module"
                               " is imported correctly.");
    // bool compile_only = !is_target_supported(target);

    // ComputeFuncHolder cf(new StatefulComputeFunc(cfi));
    ComputeFuncHolder cf(new OnlineQuantComputeFunc(
      xg, target, in_tensor_names, out_tensor_names, rt_name_, run_options
    ));
    RtModHolder rt_mod(new RuntimeModule(cf, in_tensor_names, out_tensor_names, run_options));

    return rt_mod;
  }

  // If on-the-fly quantization is not enabled, we just try to create
  //    the compute func
  if (!is_target_supported(target))
    throw std::invalid_argument("The specified runtime `" + rt_name_ + "` doesn't support the "
                                + " given target: `" + target + "`");

  ComputeFuncHolder cf = ComputeFuncFactory::GetComputeFunc(
    xg, target, in_tensor_names, out_tensor_names, rt_name_, run_options
  );
  RtModHolder rt_mod(new RuntimeModule(cf, in_tensor_names, out_tensor_names, run_options));

  return rt_mod;
}

} // namespace runtime
} // namespace pyxir
