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

#include "pyxir/runtime/compute_func_factory_impl.hpp"
#include "pyxir/opaque_func_registry.hpp"

namespace pyxir {
namespace runtime {


ComputeFuncHolder ComputeFuncFactoryImpl::get_compute_func(
  std::shared_ptr<graph::XGraph> &xg,
  const std::string &target,
  const std::vector<std::string> &in_tensor_names,
  const std::vector<std::string> &out_tensor_names,
  RunOptionsHolder const &run_options)
{
  if (!OpaqueFuncRegistry::Exists("pyxir.build_rt"))
    throw std::runtime_error("Cannot build the runtime because the "
                             " `pyxir.build_rt` opaque function is not "
                             " registered. Check if Pyxir python module"
                             " is imported correctly.");

  // If online quantization is not enabled, we just try to build the runtime
  OpaqueFunc build_rt_func = OpaqueFuncRegistry::Get("pyxir.build_rt");

  OpaqueFuncHolder rt_mod_of = std::make_shared<OpaqueFunc>(OpaqueFunc());
  build_rt_func(xg, target, rt_name_, in_tensor_names, out_tensor_names,
                rt_mod_of);

  ComputeFuncHolder cf(new OpaqueComputeFunc(rt_mod_of));

  return cf;
}

} // namespace runtime
} // namespace pyxir
