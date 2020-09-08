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
#include <vector>
#include <functional>

#include "../common/xbuffer.hpp"
#include "../common/px_stream.hpp"


namespace pyxir {
namespace runtime {

typedef void* FuncState;
typedef std::function<int(FuncState*)> AllocFuncFType;
typedef std::function<void(FuncState, 
                           std::vector<XBufferHolder> &,
                           std::vector<XBufferHolder> &)> ComputeFuncFType;
typedef std::function<void(FuncState)> ReleaseFuncFType;
typedef std::function<void(FuncState, PxOStringStream &)> SerializationFuncFType;
typedef std::function<void(FuncState*, PxIStringStream &)> DeserializationFuncFType;

struct ComputeFuncInfo {
  AllocFuncFType alloc_func;
  ComputeFuncFType compute_func;
  ReleaseFuncFType release_func;
  SerializationFuncFType serial_func;
  DeserializationFuncFType deserial_func;
};

} // namespace runtime
} // namespace pyxir