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

#include <dlfcn.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

#include "pyxir/pyxir.hpp"
#include "pyxir/ffi/str_container.hpp"
#include "pyxir/opaque_func_registry.hpp"
#include "pyxir/contrib/dpuv1.hpp"
#include "pyxir/runtime/runtime_module_factory.hpp"

namespace py = pybind11;

namespace pyxir {

  // INITIALIZATION RELATED CODE

bool py_is_initialized() { return Py_IsInitialized(); }

void PyInitializer::initialize_py() {
  // Only initialize Python interpreter if it the intepreter hasn't been
  //  initialized yet
  if (!py_is_initialized()) {
    // dlopen python library for potential issue: https://github.com/pybind/pybind11/issues/3555
    // auto py_version = exec_cmd("python3 --version");
	void* const libpython_handle = dlopen(PYTHON_LIB, RTLD_LAZY | RTLD_GLOBAL);

    // py_handle_ = dlopen("libpython3.7m.so", RTLD_LAZY | RTLD_GLOBAL);
    py::initialize_interpreter();

    py::list sys_modules = py::module::import("sys").attr("modules");
    bool px_imported = sys_modules.attr("__contains__")("pyxir").cast<bool>();
    if (!px_imported) {
      auto px = py::module::import("pyxir");
    }
  } 
}

void PyInitializer::finalize_py() {
  // Only manually finalize Python interpreter if it the intepreter hasn't been finalized yet
  if (py_is_initialized()) {
    py::finalize_interpreter();
  } 
}

PyInitializer py_initializer;

// PARTITIONING RELATED CODE

PX_API void partition(
  std::shared_ptr<graph::XGraph> xg,
  const std::vector<std::string> &targets,
  const std::string &last_layer
) {
  // Setup python port on first API call
  py_initializer.initialize_py();
  if (!OpaqueFuncRegistry::Exists("pyxir.partition"))
    throw std::runtime_error("Cannot partition XGraph because "
                             " `pyxir.partition` opaque function is not "
                             " registered. Check if Pyxir python module"
                             " is imported correctly.");
  
  OpaqueFunc partition_func = OpaqueFuncRegistry::Get("pyxir.partition");

  partition_func(xg, targets, last_layer);
}

PX_API RtModHolder build_rt(std::shared_ptr<graph::XGraph> &xg,
                            const std::string &target,
                            const std::vector<std::string> &in_tensor_names,
                            const std::vector<std::string> &out_tensor_names,
                            const std::string &runtime,
                            RunOptionsHolder const &run_options)
{
  // Setup python port on first API call
  py_initializer.initialize_py();
  return runtime::RuntimeModuleFactory::GetRuntimeModule(
    xg, target, in_tensor_names, out_tensor_names, runtime, run_options
  );
}

PX_API std::shared_ptr<graph::XGraph> load(
  const std::string &model_path, const std::string &params_path
) {
  // Setup python port on first API call
  py_initializer.initialize_py();
  if (!pyxir::OpaqueFuncRegistry::Exists("pyxir.io.load"))
    throw std::runtime_error("Cannot import ONNX model from file because"
                             " `pyxir.io.load` opaque function is"
                             " not registered");
  
  std::shared_ptr<pyxir::graph::XGraph> xg = 
    std::make_shared<pyxir::graph::XGraph>("empty_xgraph");
  
  OpaqueFunc load_func = 
    pyxir::OpaqueFuncRegistry::Get("pyxir.io.load");

  load_func(model_path, params_path, xg);

  return xg;
}

// Global variables

REGISTER_OPAQUE_FUNC("pyxir.use_dpuczdx8g_vart")
  ->set_func([](pyxir::OpaqueArgs &args) 
  {
    #if defined(USE_DPUCZDX8G_VART) || defined(USE_VART_EDGE_DPU)
      args[0]->get_str_container()->set_string("True");
    #else
      args[0]->get_str_container()->set_string("False");
    #endif
  }, std::vector<pxTypeCode>{pxStrContainerHandle});

} // pyxir
