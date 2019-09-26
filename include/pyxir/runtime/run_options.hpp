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

#include <cstdlib>
#include <memory>

namespace pyxir {
namespace runtime {

struct RunOptions {
  
  RunOptions() {
    const char *env_quant_size = std::getenv("PX_QUANT_SIZE");
    if (env_quant_size != NULL)
      nb_quant_inputs = std::atoi(env_quant_size);
  }

  // Quantization related run options
  // The default number of inputs for quantization calibration is 128
  // This can be changed by setting the 'PX_QUANT_SIZE' environment variable
  bool on_the_fly_quantization = false;
  int nb_quant_inputs = 128;

};

} // namespace runtime

typedef std::unique_ptr<runtime::RunOptions> RunOptionsHolder;

} // namespace pyxir
