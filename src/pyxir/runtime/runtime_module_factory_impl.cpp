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

#include <iostream>
#include <memory>


#include "pyxir/opaque_func_registry.hpp"
#include "pyxir/runtime/compute_func_factory.hpp"
#include "pyxir/runtime/runtime_module_factory_impl.hpp"
#include "online_quant_compute_func.hpp"


namespace pyxir {
namespace runtime {


RtModHolder DefaultRuntimeModuleFactoryImpl::get_runtime_module(
  std::shared_ptr<graph::XGraph> &xg,
  const std::string &target,
  const std::vector<std::string> &in_tensor_names,
  const std::vector<std::string> &out_tensor_names,
  std::unique_ptr<RunOptions> const &run_options)
{
  if (!OpaqueFuncRegistry::Exists("pyxir.build_rt"))
    throw std::runtime_error("Cannot build the runtime because the "
                             " `pyxir.build_rt` opaque function is not "
                             " registered. Check if Pyxir python module"
                             " is imported correctly.");

  // If the on-the-fly quantization option is enabled, we create a compute function
  //  that performs quantization calibration on the first N inputs (and computes
  //  actual results on CPU) and afterwards switches to hardware acceleration
  if (run_options && run_options->on_the_fly_quantization
      && run_options->nb_quant_inputs > 0)
  {
    if (!OpaqueFuncRegistry::Exists("pyxir.quantize"))
      throw std::runtime_error("Cannot runtime with on-the-fly quantization"
                               " because the "
                               " `pyxir.quantize` opaque function is not "
                               " registered. Check if Pyxir python module"
                               " is imported correctly.");
    int nb_quant_inputs = run_options->nb_quant_inputs;

    ComputeFuncInfo cfi;
    cfi.alloc_func = [this, &xg, target, &in_tensor_names, &out_tensor_names, 
                      nb_quant_inputs](FuncState *state) 
    {
      auto *online_quant_cf = new OnlineQuantComputeFunc(
        xg, target, in_tensor_names, out_tensor_names, rt_name_, nb_quant_inputs
      );
      *state = online_quant_cf;
      return 0;
    };

    cfi.release_func = [](FuncState state)
    {
      if (state)
        delete reinterpret_cast<OnlineQuantComputeFunc*>(state);
    };

    cfi.compute_func = [](FuncState state, 
                          std::vector<pyxir::XBufferHolder> &in_tensors,
                          std::vector<pyxir::XBufferHolder> &out_tensors)
    {
      OnlineQuantComputeFunc* oqcf =
        reinterpret_cast<OnlineQuantComputeFunc*>(state);
      (*oqcf)(in_tensors, out_tensors);
    };

    ComputeFuncHolder cf(new StatefulComputeFunc(cfi));
    RtModHolder rt_mod(new RuntimeModule(cf));

    return rt_mod;
  }

  // If on-the-fly quantization is not enabled, we just try to create
  //    the compute func
  ComputeFuncHolder cf = ComputeFuncFactory::GetComputeFunc(
    xg, target, in_tensor_names, out_tensor_names, rt_name_
  );
  RtModHolder rt_mod(new RuntimeModule(cf));

  return rt_mod;
}

} // namespace runtime
} // namespace pyxir
