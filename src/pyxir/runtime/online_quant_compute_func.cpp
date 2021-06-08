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

#include "pyxir/ffi/str_container.hpp"
#include "pyxir/runtime/runtime_module_factory.hpp"
#include "pyxir/runtime/compute_func_factory.hpp"
#include "pyxir/runtime/compute_func_registry.hpp"
#include "pyxir/runtime/online_quant_compute_func.hpp"


namespace pyxir {
namespace runtime {

OnlineQuantComputeFunc::OnlineQuantComputeFunc(
    XGraphHolder &xg, const std::string &target,
    const std::vector<std::string> &in_tensor_names,
    const std::vector<std::string> &out_tensor_names,
    const std::string &runtime, RunOptionsHolder &run_options)
    : xg_(xg), target_(target), in_tensor_names_(in_tensor_names),
      out_tensor_names_(out_tensor_names), runtime_(runtime),
      run_options_(run_options) {
  init();
}

void OnlineQuantComputeFunc::init()
{
  OpaqueFunc build_online_quant_rt_func =
    OpaqueFuncRegistry::Get("pyxir.build_online_quant_rt");

  is_target_supported_ = RuntimeModuleFactory::SupportsTarget(runtime_, target_);

  // If PX_BUILD_DIR environment variable is set, we overwrite the run_options build_dir
  //  member variable
  const char *env_tmp = std::getenv("PX_BUILD_DIR");
  if (env_tmp != NULL) {
    run_options_->is_prebuilt = true;
    run_options_->build_dir = env_tmp;
  }

  if (run_options_->is_prebuilt && is_target_supported_) {
    cf_ = ComputeFuncFactory::GetComputeFunc(
      xg_, target_, in_tensor_names_, out_tensor_names_, runtime_, run_options_
    ); 

    // Make sure quantization function doesn't get called
    count_ = run_options_->nb_quant_inputs + 1;
  } else if (run_options_->is_prebuilt && !is_target_supported_) {
    // Do not create compute function
    pxInfo("Cross compiling for different target device: " + target_);
  } else {   
    quant_of_ = std::make_shared<OpaqueFunc>(OpaqueFunc());
    OpaqueFuncHolder rt_func_of = std::make_shared<OpaqueFunc>(OpaqueFunc());
    build_online_quant_rt_func(xg_, target_, runtime_, in_tensor_names_,
      out_tensor_names_, run_options_->build_dir, run_options_->work_dir,
      quant_of_, rt_func_of);
	  cf_ = ComputeFuncHolder(new OpaqueComputeFunc(rt_func_of));
  }
}

void OnlineQuantComputeFunc::operator()(
  std::vector<XBufferHolder> &in_tensors,
  std::vector<XBufferHolder> &out_tensors)
{
  if (cf_) {
    (*cf_)(in_tensors, out_tensors);
    ++count_;
  } else if (!is_target_supported_) {
    throw std::runtime_error("Trying to run on unsupported target: " + target_);
  } else {
    throw std::runtime_error("Trying to run uninstantiated compute function");
  }

  if (count_ == run_options_->nb_quant_inputs) {
    // Call quantization function
    OpaqueArgs args = OpaqueArgs();
    quant_of_->call(args);

    if (!is_target_supported_) {
      // Just do cross compilation
      if (!OpaqueFuncRegistry::Exists("pyxir.compile"))
        throw std::runtime_error("Cannot compile the Vitis-AI compute func because the "
                                " `pyxir.compile` opaque function is not "
                                " registered. Check if Pyxir python module"
                                " is imported correctly.");

      XGraphHolder scheduled_xg =
        std::make_shared<pyxir::graph::XGraph>("scheduled_xgraph"); 

      OpaqueFunc compile_func = OpaqueFuncRegistry::Get("pyxir.compile");
      
      compile_func(xg_, target_, in_tensor_names_, out_tensor_names_,
                   run_options_->build_dir, run_options_->work_dir, scheduled_xg);

      pxInfo("Not switching to specified runtime: `" + runtime_ + "` after on-the-fly" +
             " quantization as the model is compiled for a different target device.");
    } else {
      cf_ = ComputeFuncFactory::GetComputeFunc(
        xg_, target_, in_tensor_names_, out_tensor_names_, runtime_, run_options_
      );
    }

    // Debug runtime and target!! DEBUG ONLY
    const char *px_debug_runtime_flag = std::getenv("PX_DEBUG_RUNTIME");
    const char *px_debug_target_flag = std::getenv("PX_DEBUG_TARGET");
    if (px_debug_runtime_flag != NULL && px_debug_target_flag != NULL) {
      std::string px_debug_runtime = std::string(px_debug_runtime_flag);
      std::string px_debug_target = std::string(px_debug_target_flag);
      pxWarning("Switching to debug runtime: " + px_debug_runtime + ", with target: " + px_debug_target);
      cf_ = ComputeFuncFactory::GetComputeFunc(
        xg_, px_debug_target, in_tensor_names_, out_tensor_names_, px_debug_runtime, run_options_
      );
    }
    
    // The final runtime has been built now
    run_options_->is_prebuilt = true;
    // We possibly save the runtime module using a callback function
    //  Currently necessary for ONNX Runtime flow. TODO: remove this requirement
    if (run_options_ && !run_options_->export_runtime_module_path.empty()) {
      if (!rt_mod_save_callback_)
        throw std::runtime_error("Trying to export cross compiled runtime module but"
                                 " compute save function was initialized uncorrectly");
      rt_mod_save_callback_(run_options_->export_runtime_module_path);
    }
  }
}

void OnlineQuantComputeFunc::serialize_px(PxOStringStream &pstream)
{
  // Serialize XGraph
  pyxir::write(xg_, pstream);

  // Serialize target and runtime
  pstream.write(target_);
  pstream.write(runtime_);

  // Serialize in and out tensor names
  pstream.write(in_tensor_names_.size());
  for (auto & it : in_tensor_names_) {
    pstream.write(it);
  }
  pstream.write(out_tensor_names_.size());
  for (auto & ot : out_tensor_names_) {
    pstream.write(ot);
  }

  // Serialize run options
  run_options_->serialize_px(pstream);
  
  // Serialize build directory
  pyxir::OpaqueFunc serialize_dir =
    pyxir::OpaqueFuncRegistry::Get("pyxir.io.serialize_dir");
  BytesContainerHolder zip_bytes_c = BytesContainerHolder(new BytesContainer());
  serialize_dir(run_options_->build_dir, zip_bytes_c);
  pstream.write(zip_bytes_c->get_string());
}

void OnlineQuantComputeFunc::deserialize_px(PxIStringStream &pstream)
{
  // Deserialize XGraph
  pyxir::read(xg_, pstream);

  // Deserealize target and runtime
  pstream.read(target_);
  pstream.read(runtime_);

  // Deserialize in and out tensor names
  int it_size;
  pstream.read(it_size);
  for (int i = 0; i < it_size; ++i) {
    std::string it_name;
    pstream.read(it_name);
    in_tensor_names_.push_back(it_name);
  }
  int ot_size;
  pstream.read(ot_size);
  for (int i = 0; i < ot_size; ++i) {
    std::string ot_name;
    pstream.read(ot_name);
    out_tensor_names_.push_back(ot_name);
  }

  // Deserialize run options
  run_options_->deserialize_px(pstream);

  // Deserialize build directory
  std::string zip_str;
  pstream.read(zip_str);
  pyxir::OpaqueFunc deserialize_dir =
    pyxir::OpaqueFuncRegistry::Get("pyxir.io.deserialize_dir");
  deserialize_dir(run_options_->build_dir, zip_str);

  // Initialize
  init();
}

OnlineQuantComputeFunc::~OnlineQuantComputeFunc()
{
  // Remove work and build directories
  if (!run_options_->work_dir.empty() && pyxir::is_dir(run_options_->work_dir))
    pyxir::rmrf(run_options_->work_dir);
  if (!run_options_->build_dir.empty() && pyxir::is_dir(run_options_->build_dir))
    pyxir::rmrf(run_options_->build_dir);
}

REGISTER_COMPUTE_FUNC_TYPE("online_quant_compute_func")
  .set_factory_func([]() -> ComputeFuncHolder {
    ComputeFuncHolder cf(new OnlineQuantComputeFunc());
    return cf;
  });

} // namespace runtime
} // namespace pyxir
