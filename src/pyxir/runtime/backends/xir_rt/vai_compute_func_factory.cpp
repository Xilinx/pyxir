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

#include "vai_compute_func_factory.hpp"
#include "pyxir/opaque_func_registry.hpp"
#include "pyxir/graph/xgraph.hpp"
#include "vai_compute_func.hpp"


namespace pyxir {
namespace runtime {
namespace vai_rt {


ComputeFuncHolder VaiComputeFuncFactoryImpl::get_compute_func(
  std::shared_ptr<graph::XGraph> &xg,
  const std::string &target,
  const std::vector<std::string> &in_tensor_names,
  const std::vector<std::string> &out_tensor_names,
  RunOptionsHolder const &run_options)
{ 
  XGraphHolder scheduled_xg =
      std::make_shared<pyxir::graph::XGraph>("scheduled_xgraph"); 
 
  // If PX_BUILD_DIR environment variable is set, we load the scheduled xgraph from
  // 	file
  // const char *env_build_dir = std::getenv("PX_BUILD_DIR");
  if (run_options->is_prebuilt) {
    std::string build_dir = run_options->build_dir;
    if (!OpaqueFuncRegistry::Exists("pyxir.io.load_scheduled_xgraph_from_meta"))
      throw std::runtime_error("Cannot build the Vitis-AI compute func because the "
                               " `pyxir.io.load_scheduled_xgraph_from_meta`"
                               " opaque function is not "
                               " registered. Check if Pyxir python module"
                               " is imported correctly.");
    OpaqueFunc load_func =
      OpaqueFuncRegistry::Get("pyxir.io.load_scheduled_xgraph_from_meta");

    load_func(build_dir, scheduled_xg);
  } else {
    if (!OpaqueFuncRegistry::Exists("pyxir.compile"))
      throw std::runtime_error("Cannot build the Vitis-AI compute func because the "
                               " `pyxir.compile` opaque function is not "
                               " registered. Check if Pyxir python module"
                               " is imported correctly.");

    OpaqueFunc compile_func = OpaqueFuncRegistry::Get("pyxir.compile");
    
    std::string build_dir = run_options->build_dir;
    std::string work_dir = run_options->work_dir;
    compile_func(xg, target, in_tensor_names, out_tensor_names, build_dir,
                 work_dir, scheduled_xg);
  }

  // Create stateful compute function
  ComputeFuncInfo cfi;
  cfi.alloc_func = [this, &scheduled_xg, target, &in_tensor_names,
                    &out_tensor_names, &run_options](FuncState *state) 
  {
    auto *vai_cf = new VaiComputeFunc(
      scheduled_xg, target, in_tensor_names, out_tensor_names, run_options->build_dir
    );
    *state = vai_cf;
    return 0;
  };

  cfi.release_func = [](FuncState state)
  {
    if (state)
      delete reinterpret_cast<VaiComputeFunc*>(state);
  };

  cfi.compute_func = [](FuncState state, 
                        std::vector<pyxir::XBufferHolder> &in_tensors,
                        std::vector<pyxir::XBufferHolder> &out_tensors)
  {
    VaiComputeFunc* vai_cf =
      reinterpret_cast<VaiComputeFunc*>(state);
    (*vai_cf)(in_tensors, out_tensors);
  };

  ComputeFuncHolder cf(new StatefulComputeFunc(cfi));

  return cf;
}

} // vai_rt
} // namespace runtime
} // namespace pyxir
