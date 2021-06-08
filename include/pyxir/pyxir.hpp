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

#include <memory>

#include "graph/xgraph.hpp"
#include "io/io.hpp"
#include "pyxir_api.hpp"
#include "runtime/constants.hpp"
#include "runtime/run_options.hpp"
#include "runtime/runtime_module_factory.hpp"

namespace pyxir {

/**
 * @brief Partition the provided XGraph for the given target(s)
 * @param xg The XGraph to be partitioned
 * @param targets The target(s) for which to partition the XGraph (only
 *  one target is supported at the moment)
 * @param last_layer The last layer uptil which to do partitioning of the 
 *  XGraph
 */
PX_API void partition(std::shared_ptr<graph::XGraph> xg,
                      const std::vector<std::string> &targets,
                      const std::string &last_layer = "");

/**
 * @brief Build a runtime module for executing the provided XGraph model
 * @param xg The XGraph model to be executed
 * @param target The target for executing the XGraph model
 * @param in_tensor_names The names of the input tensors that will be
 *  provided in the same order
 * @param out_tensor_names The names of the output tensors that will be
 *  provided in the same order
 * @param runtime (optional) The runtime to be used for executing the model
 * @param run_options (optional) The specified run options, e.g. whether online 
 *  quantization should be enabled
 * @returns A runtime module to can be used for execution of the provided
 *  XGraph
 */
PX_API RtModHolder build_rt(std::shared_ptr<graph::XGraph> &xg,
                            const std::string &target,
                            const std::vector<std::string> &in_tensor_names,
                            const std::vector<std::string> &out_tensor_names,
                            const std::string &runtime = runtime::pxCpuTfRuntimeModule,
                            RunOptionsHolder const &run_options = nullptr);


/**
 * @brief Load an XGraph model from file
 * @param model_path The path to the model graph file
 * @param params_path The path to the model params file
 * @returns A shared pointer to the loaded XGraph object
 */
PX_API std::shared_ptr<graph::XGraph> load(const std::string &model_path,
                                           const std::string &params_path);


/**
 * @brief Return whether the Python interpreter is initialized (for
 *  internal use)
 */
PX_API bool py_is_initialized();

} // pyxir

