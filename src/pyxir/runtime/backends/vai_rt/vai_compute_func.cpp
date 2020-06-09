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
#include <cassert>
#include <algorithm>
#include <chrono>

#include "pyxir/common/util.hpp"
#include "vai_compute_func.hpp"

void vaiDebugMsg(const char * msg, const char *funcname,
                 const char *fname, int lineno) {
  std::cout << "VITISAI(" << funcname << "): " << msg << " (" 
    << fname << ":" << lineno << ")" << std::endl;
}

namespace pyxir {
namespace runtime {
namespace vai_rt {

VaiComputeFunc::VaiComputeFunc(
  XGraphHolder &xg,
  const std::string &target,
  const std::vector<std::string> &in_tensor_names,
  const std::vector<std::string> &out_tensor_names)
  : xg_(xg), target_(target)
{
  XLayerHolder dpu_X;
  
  // Check whether we can execute all layers of this XGraph and find
  //  the DPU layer
  for (std::string &xl_name : xg->get_layer_names())
  {
    XLayerHolder X = xg->get(xl_name);
    if (!is_op_supported(X->xtype[0]))
      throw std::invalid_argument("VAI Runtime got unsupported operation of"
                                  " type: " + X->xtype[0]);
    else if (X->xtype[0] == "DPUV1" || X->xtype[0] == "DPUV2")
      dpu_X = X;
  }

  std::vector<std::string> dpu_in_tensor_names = dpu_X->bottoms;
  std::vector<std::string> dpu_internal_in_tensor_names
    = dpu_X->get_attr("input_names").get_strings();
  
  // assert(in_tensor_names_.size() == dpu_in_tensor_names.size());
  assert(dpu_in_tensor_names.size() == dpu_internal_in_tensor_names.size());
  for (const std::string &itn : in_tensor_names) {
    if (std::find(dpu_in_tensor_names.begin(), dpu_in_tensor_names.end(),
        pyxir::stringify(itn)) != dpu_in_tensor_names.end()) {
      std::vector<std::string>::iterator iter = std::find_if(
        dpu_in_tensor_names.begin(),
        dpu_in_tensor_names.end(),
        [itn](const std::string &elem) { return elem == pyxir::stringify(itn); });
      size_t index = std::distance(dpu_in_tensor_names.begin(), iter);
      
      if (index < dpu_in_tensor_names.size()) {
        in_tensor_names_.push_back(dpu_internal_in_tensor_names[index]);
  	  } else {
        throw std::runtime_error("Could not find input tensor '" 
          + pyxir::stringify(itn) 
          + "' inside Vitis-AI accelerator input tensors");
      }
    }
  }

  std::vector<std::string> dpu_out_tensor_names = dpu_X->tops;
  // assert(out_tensor_names_.size() == dpu_out_tensor_names.size());
  for (const std::string &otn : out_tensor_names) {
    if (std::find(dpu_X->tops.begin(), dpu_X->tops.end(), pyxir::stringify(otn))
        != dpu_X->tops.end()) {
      out_tensor_names_.push_back(pyxir::stringify(otn));
  	} else {
      throw std::runtime_error("Could not find output tensor '" 
        + pyxir::stringify(otn) 
        + "' inside Vitis-AI accelerator input tensors");
    }
  }
  
  std::unordered_map<std::string, std::string> rt_in_map = 
    dpu_X->get_attr("rt_in_map").get_map_str_str();
  std::unordered_map<std::string, std::string> rt_out_map = 
    dpu_X->get_attr("rt_out_map").get_map_str_str();
  // assert(in_tensor_names_.size() == rt_in_map.size());
  // assert(out_tensor_names_.size() == rt_out_map.size());
  
  vaiDebug("Before DpuRunner init");
  // Setup DPU runner
  // If PX_BUILD_DIR environment variable is set, we use that directory
  //   to setup the DpuRunner
  std::string model_path;
  const char *env_build_dir = std::getenv("PX_BUILD_DIR");
  if (env_build_dir != NULL) {
    model_path = env_build_dir;
  } else {
    model_path = dpu_X->get_attr("work_dir").get_string();
  }

  auto dpu_runners = vitis::ai::DpuRunner::create_dpu_runner(model_path);
  dpu_runner_ = std::move(dpu_runners[0]);

  if(dpu_runner_->get_tensor_format() != vitis::ai::DpuRunner::TensorFormat::NCHW) {
    vaiDebug("Invalid tensor format NHWC");
  }
  vaiDebug("After DpuRunner init");
  
  dpu_runner_in_tensors_ = dpu_runner_->get_input_tensors();
  dpu_runner_out_tensors_ = dpu_runner_->get_output_tensors();
  assert(dpu_runner_in_tensors_.size() == dpu_in_tensor_names.size());
  assert(dpu_runner_out_tensors_.size() == dpu_out_tensor_names.size());

  std::vector<std::string> dpu_runner_in_tensor_names;
  std::transform(dpu_runner_in_tensors_.begin(), dpu_runner_in_tensors_.end(),
   std::back_inserter(dpu_runner_in_tensor_names),
   [](vitis::ai::Tensor* t) -> const std::string & { return t->get_name(); });

  std::vector<std::string> dpu_runner_out_tensor_names;
  std::transform(dpu_runner_out_tensors_.begin(), dpu_runner_out_tensors_.end(),
   std::back_inserter(dpu_runner_out_tensor_names),
   [](vitis::ai::Tensor* t) -> const std::string & { return t->get_name(); });

  std::vector<std::string> rt_in_names;
  std::transform(in_tensor_names_.begin(), in_tensor_names_.end(),
   std::back_inserter(rt_in_names),
   [&rt_in_map](const std::string &elem)
   -> const std::string & { return rt_in_map[elem]; });

  for (int i = 0; i < in_tensor_names_.size(); ++i)
  {
    std::string dpu_in_name = dpu_runner_in_tensor_names[i];

    std::vector<std::string>::iterator iter = std::find_if(
      rt_in_names.begin(), rt_in_names.end(),
      [dpu_in_name](const std::string &elem) { return elem == dpu_in_name; });
    size_t index = std::distance(rt_in_names.begin(), iter);
    if (index == rt_in_names.size())
    {
      throw std::runtime_error("DPU in tensor: " + dpu_in_name 
        + " not found in model runtime naming map.");
    }
    else 
    {
      in_tensor_order_.push_back(index);
    }
  }

  std::vector<std::string> rt_out_names;
  std::transform(out_tensor_names_.begin(), out_tensor_names_.end(),
   std::back_inserter(rt_out_names),
   [&rt_out_map](const std::string &elem)
   -> const std::string & { return rt_out_map[elem]; });

  for (int i = 0; i < out_tensor_names_.size(); ++i)
  {
    std::string dpu_out_name = dpu_runner_out_tensor_names[i];
    
    std::vector<std::string>::iterator iter = std::find_if(
      rt_out_names.begin(),
      rt_out_names.end(),
      [dpu_out_name](const std::string &elem) { return elem == dpu_out_name; });
    size_t index = std::distance(rt_out_names.begin(), iter);
    if (index == rt_out_names.size())
    {
      throw std::runtime_error("DPU out tensor: " + dpu_out_name 
        + " not found in model runtime naming map.");
    }
    else 
    {
      out_tensor_order_.push_back(index);
    }
  }

  vaiDebug("Inside Initialize print in/out maps");
}

void VaiComputeFunc::operator()(
  std::vector<XBufferHolder> &in_tensors,
  std::vector<XBufferHolder> &out_tensors)
{
  vaiDebug("Inside VaiComputeFunc::()");

  std::vector<vitis::ai::CpuFlatTensorBuffer> inputs_cpu, outputs_cpu;
  std::vector<vitis::ai::TensorBuffer*> in_buffer, out_buffer;
  std::vector<std::shared_ptr<vitis::ai::Tensor>> batch_tensors;

  for (ssize_t i = 0; i < in_tensor_names_.size(); ++i)
  {
    batch_tensors.push_back(
      std::shared_ptr<vitis::ai::Tensor>(
        new vitis::ai::Tensor(dpu_runner_in_tensors_[i]->get_name(),
                              dpu_runner_in_tensors_[i]->get_dims(),
                              dpu_runner_in_tensors_[i]->get_data_type())
      )
    );

    inputs_cpu.push_back(
      vitis::ai::CpuFlatTensorBuffer(
        in_tensors[in_tensor_order_[i]]->data,
        batch_tensors.back().get())
    );
  }

  for (ssize_t i = 0; i < out_tensor_names_.size(); ++i)
  {
    batch_tensors.push_back(
      std::shared_ptr<vitis::ai::Tensor>(
        new vitis::ai::Tensor(dpu_runner_out_tensors_[i]->get_name(),
                              dpu_runner_out_tensors_[i]->get_dims(),
                              dpu_runner_out_tensors_[i]->get_data_type())
      )
    );
    outputs_cpu.push_back(
      vitis::ai::CpuFlatTensorBuffer(
        out_tensors[out_tensor_order_[i]]->data,
        batch_tensors.back().get())
    );
  }

  for(size_t i = 0; i < dpu_runner_in_tensors_.size(); ++i) {
    in_buffer.push_back(&inputs_cpu[i]);
  }

  for(size_t i = 0; i < dpu_runner_out_tensors_.size(); ++i) {
    out_buffer.push_back(&outputs_cpu[i]);
  }

  /*run*/
  auto start = std::chrono::high_resolution_clock::now();
  auto job_id = dpu_runner_->execute_async(in_buffer, out_buffer);
  auto stop1 = std::chrono::high_resolution_clock::now();
  dpu_runner_->wait(job_id.first,-1);
  auto stop2 = std::chrono::high_resolution_clock::now();


  // std::cout << "exec_async Time: " << std::chrono::duration_cast<std::chrono::microseconds>(stop1-start).count() << std::endl;
  // std::cout << "wait Time: " << std::chrono::duration_cast<std::chrono::microseconds>(stop2-stop1).count() << std::endl;
  // std::cout << "total Time: " << std::chrono::duration_cast<std::chrono::microseconds>(stop2-start).count() << std::endl;

}

} // vai_rt
} // namespace runtime
} // namespace pyxir
