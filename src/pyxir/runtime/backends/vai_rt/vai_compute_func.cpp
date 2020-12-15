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
#include <unordered_map>
#include <chrono>

#include "vai_compute_func.hpp"
#include "dpu_func.hpp"

#include "pyxir/common/util.hpp"
#include "../cpu/input.hpp"
#include "../cpu/transpose.hpp"
#include "../cpu/tuple_get_item.hpp"
#include "../cpu/tuple.hpp"


namespace pyxir {
namespace runtime {
namespace vai_rt {

VaiComputeFunc::VaiComputeFunc(
  XGraphHolder &xg,
  const std::string &target,
  const std::vector<std::string> &in_tensor_names,
  const std::vector<std::string> &out_tensor_names,
  const std::string &build_dir)
  : xg_(xg), target_(target), build_dir_(build_dir)
{
  pxDebug("Initialize VaiComputeFunc");

  for (const std::string &itn : in_tensor_names)
    in_tensor_names_.push_back(pyxir::stringify(itn));

  for (const std::string &otn : out_tensor_names)
    out_tensor_names_.push_back(pyxir::stringify(otn));
  
  // Check whether we can execute all layers of this XGraph and find
  //  the DPU layer
  for (std::string &xl_name : xg->get_layer_names())
  {
    XLayerHolder X = xg->get(xl_name);

    if (X->xtype[0] == "DPU" || X->xtype[0] == "DPUV1" || X->xtype[0] == "DPUV2") {
      std::unique_ptr<KernelFunc> dpu_func(new DpuFunc(X, build_dir_)); 
      kernel_funcs_.push_back(std::move(dpu_func));
    } else if (X->xtype[0] == "Input") {
      // Skip input as it's an identity operation
      std::unique_ptr<KernelFunc> input_func(new cpu::InputFunc(X));
      kernel_funcs_.push_back(std::move(input_func));
    } else if (X->xtype[0] == "TupleGetItem") {
      std::unique_ptr<KernelFunc> tgi_func(new cpu::TupleGetItemFunc(X));
      kernel_funcs_.push_back(std::move(tgi_func));
    } else if (X->xtype[0] == "Tuple") {
      std::unique_ptr<KernelFunc> tuple_func(new cpu::TupleFunc(X));
      kernel_funcs_.push_back(std::move(tuple_func));
    } else if (X->xtype[0] == "Transpose") {
      std::unique_ptr<KernelFunc> transpose_func(new cpu::TransposeFunc(X));
      kernel_funcs_.push_back(std::move(transpose_func));
    } else {
      throw std::invalid_argument("VAI Runtime got unsupported operation of"
                                  " type: " + X->xtype[0]);
    }
    Xs_.push_back(X);
    // For timing tracking
    total_kernel_times_.push_back(0);
  }
}

VaiComputeFunc::~VaiComputeFunc() {
  if (is_verbose()) {
    std::cout << "---------------------" << std::endl;
    std::cout << "PX VAI COMPUTE FUNC TIMINGS: " << std::endl;
    std::cout << "Total compute time: " << std::to_string(total_compute_time_) << std::endl;
    for (int i = 0; i < Xs_.size(); ++i) {
      std::cout << "Kernel " << std::to_string(i) << " time: " <<
        std::to_string(total_kernel_times_[i]) << std::endl;
    }
    std::cout << "---------------------" << std::endl;
  }
}

void VaiComputeFunc::operator()(
  std::vector<XBufferHolder> &in_tensors,
  std::vector<XBufferHolder> &out_tensors)
{
  auto start_vai = std::chrono::high_resolution_clock::now();
  pxDebug("Inside VaiComputeFunc::()");

  std::unordered_map<std::string, std::vector<XBufferHolder>> int_res;

  for (int i = 0; i < in_tensors.size(); ++i) {
    std::vector<XBufferHolder> in_v{in_tensors[i]};
    int_res[in_tensor_names_[i]] = std::move(in_v);
  }

  for (int i = 0; i < out_tensors.size(); ++i) {
    std::vector<XBufferHolder> out_v{out_tensors[i]};
    int_res[out_tensor_names_[i]] = std::move(out_v);
  }

  std::vector<XBufferHolder> dpu_in;
  std::vector<XBufferHolder> dpu_out;

  auto stop_init = std::chrono::high_resolution_clock::now();
  std::chrono::microseconds duration_init = std::chrono::duration_cast<std::chrono::microseconds>(stop_init-start_vai);
  pxDebug(("Init time: " + std::to_string(duration_init.count())).c_str());

  for (int i = 0; i < Xs_.size(); ++i) {
    auto start_k_begin = std::chrono::high_resolution_clock::now();
    dpu_in.clear();
    dpu_out.clear();
    auto stop_clear = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration_clear = std::chrono::duration_cast<std::chrono::microseconds>(stop_clear-start_k_begin);
    pxDebug(("Clear time: " + std::to_string(duration_clear.count())).c_str());

    XLayerHolder &X = Xs_[i];

    if (X->bottoms.empty())
      dpu_in.insert(dpu_in.end(), int_res[X->name].begin(),  int_res[X->name].end());

    for (const std::string &itn : X->bottoms)
      dpu_in.insert(dpu_in.end(), int_res[itn].begin(),  int_res[itn].end());

    // for (const std::string &otn : X->tops) {
    const std::string &otn = X->name;
    if (int_res.find(otn) != int_res.end())
      dpu_out.insert(dpu_out.end(), int_res[otn].begin(), int_res[otn].end());
    
    auto start_k = std::chrono::high_resolution_clock::now();
    kernel_funcs_[i]->operator()(dpu_in, dpu_out);
    auto stop_k = std::chrono::high_resolution_clock::now();

    std::chrono::microseconds duration_kernel = std::chrono::duration_cast<std::chrono::microseconds>(stop_k-start_k);
    pxDebug(("Kernel time: " + std::to_string(duration_kernel.count())).c_str());
    std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_k-start_k_begin);
    pxDebug(("Time: " + std::to_string(duration.count())).c_str());
    total_kernel_times_[i] += duration.count();

    int_res[otn] = dpu_out;
  }

  auto stop = std::chrono::high_resolution_clock::now();

  std::chrono::microseconds vai_compute_time = std::chrono::duration_cast<std::chrono::microseconds>(stop-start_vai);
  total_compute_time_ += vai_compute_time.count();
  pxDebug(("Vai Compute Func Time: " + std::to_string(vai_compute_time.count())).c_str());
}

} // vai_rt
} // namespace runtime
} // namespace pyxir
