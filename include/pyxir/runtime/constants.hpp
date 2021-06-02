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

#include <string>
#include <vector>

namespace pyxir {
namespace runtime {

const std::string pxCpuTfRuntimeModule = "cpu-tf";
const std::string pxCpuNpRuntimeModule = "cpu-np";
const std::string pxCpuRuntimeModule = "cpu";
const std::string pxDecentQSimRuntimeModule = "decentq-sim";
const std::string pxVaiRuntimeModule = "vai";
const std::vector<std::string> cpuTargets {"cpu"};

#ifdef USE_VAI_RT_DPUCADX8G
const std::vector<std::string> vaiTargets {"DPUCADX8G", "dpuv1"};
#elif defined(USE_VAI_RT_DPUCZDX8G)
const std::vector<std::string> vaiTargets {"DPUCZDX8G-zcu104", "DPUCZDX8G-zcu102", "DPUCZDX8G-ultra96", "DPUCZDX8G-som",
                                           "dpuv2-zcu104", "dpuv2-zcu102", "dpuv2-ultra96", "dpuv2-som"};
#elif defined(USE_VAI_RT_DPUCAHX8H)
const std::vector<std::string> vaiTargets {"DPUCAHX8H-u50", "DPUCAHX8H-u280"};

#elif defined(USE_VART_CLOUD_DPU)
const std::vector<std::string> vaiTargets {"DPUCAHX8H-u50", "DPUCAHX8H-u280", "DPUCADF8H"};

#else
const std::vector<std::string> vaiTargets {};
#endif

} // namespace runtime
} // namespace pyxir
