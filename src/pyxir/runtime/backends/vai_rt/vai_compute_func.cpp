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

#include "pyxir/common/util.hpp"
#include "vai_compute_func.hpp"
#include "dpu_func.hpp"

void vaiDebugMsg(const char * msg, const char *funcname,
                 const char *fname, int lineno) {
  std::cout << "PYXIR(VITISAI(" << funcname << "): " << msg << " (" 
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
  // XLayerHolder dpu_X;
  
  // Check whether we can execute all layers of this XGraph and find
  //  the DPU layer
  for (std::string &xl_name : xg->get_layer_names())
  {
    XLayerHolder X = xg->get(xl_name);
    if (!is_op_supported(X->xtype[0]))
      throw std::invalid_argument("VAI Runtime got unsupported operation of"
                                  " type: " + X->xtype[0]);
    else if (X->xtype[0] == "DPUV1" || X->xtype[0] == "DPUV2")
      dpu_X_ = X;
  }

  for (const std::string &itn : in_tensor_names)
    in_tensor_names_.push_back(pyxir::stringify(itn));

  for (const std::string &otn : out_tensor_names)
    out_tensor_names_.push_back(pyxir::stringify(otn));

  dpu_func_ = DpuFunc(dpu_X_);
}

void VaiComputeFunc::operator()(
  std::vector<XBufferHolder> &in_tensors,
  std::vector<XBufferHolder> &out_tensors)
{
  vaiDebug("Inside VaiComputeFunc::()");
  auto start_vai = std::chrono::high_resolution_clock::now();

  std::unordered_map<std::string, XBufferHolder> int_res;

  for (int i = 0; i < in_tensors.size(); ++i) {
    std::string itn = in_tensor_names_[i];
    int_res[itn] = in_tensors[i];
  }

  for (int i = 0; i < out_tensors.size(); ++i) {
    std::string otn = out_tensor_names_[i];
    int_res[otn] = out_tensors[i];
  }

  std::vector<std::string> dpu_in_tensor_names = dpu_X_->bottoms;
  std::vector<std::string> dpu_out_tensor_names = dpu_X_->tops;

  std::vector<XBufferHolder> dpu_in;
  std::vector<XBufferHolder> dpu_out;

  for (const std::string &itn : dpu_in_tensor_names)
    dpu_in.push_back(int_res[itn]);

  for (const std::string &otn : dpu_out_tensor_names)
    dpu_out.push_back(int_res[otn]);

  auto start = std::chrono::high_resolution_clock::now();
  dpu_func_(dpu_in, dpu_out);
  auto stop = std::chrono::high_resolution_clock::now();

  std::cout << "Dpu Time: " << std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count() << std::endl;
  std::cout << "Vai Compute Func Time: " << std::chrono::duration_cast<std::chrono::microseconds>(stop-start_vai).count() << std::endl;

}

} // vai_rt
} // namespace runtime
} // namespace pyxir
