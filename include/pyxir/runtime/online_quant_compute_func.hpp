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

#include "../common/xbuffer.hpp"
#include "../graph/xgraph.hpp"
#include "../io/io.hpp"
#include "../opaque_func_registry.hpp"
#include "../runtime/compute_func.hpp"
#include "../runtime/run_options.hpp"

namespace pyxir {
namespace runtime {

class OnlineQuantComputeFunc : public IComputeFunc {

  public:
    OnlineQuantComputeFunc() {
      xg_ = std::make_shared<pyxir::graph::XGraph>("");
      run_options_ = RunOptionsHolder(new RunOptions());
    }

    OnlineQuantComputeFunc(const OnlineQuantComputeFunc &other)
        : xg_(other.xg_), target_(other.target_),
          in_tensor_names_(other.in_tensor_names_),
          out_tensor_names_(other.out_tensor_names_), runtime_(other.runtime_),
          run_options_(other.run_options_) {
      init();
    }

    OnlineQuantComputeFunc(OnlineQuantComputeFunc &&other)
        : xg_(other.xg_), target_(other.target_),
          in_tensor_names_(other.in_tensor_names_),
          out_tensor_names_(other.out_tensor_names_), runtime_(other.runtime_),
          run_options_(other.run_options_), count_(other.count_),
          is_target_supported_(other.is_target_supported_),
          cf_(std::move(other.cf_)), quant_of_(other.quant_of_) {}

    OnlineQuantComputeFunc(XGraphHolder &xg, const std::string &target,
                           const std::vector<std::string> &in_tensor_names,
                           const std::vector<std::string> &out_tensor_names,
                           const std::string &runtime,
                           RunOptionsHolder &run_options);
    ~OnlineQuantComputeFunc();

    /**
     * @brief Returns this function type
     */
    std::string get_type() { return "online_quant_compute_func"; }

    /**
     * @brief Initialize the OnlineQuantComputeFunc after instantiation or 
     *        deserialization
     */
    void init();

    /**
     * @brief Main runtime function for executing this compute function
     */
    void operator()(std::vector<XBufferHolder> &in_tensors,
                    std::vector<XBufferHolder> &out_tensors) override;

    /**
     * @brief Serialize this function
     */
    void serialize_px(PxOStringStream &pstream) override;

    /**
     * @brief Deserialize this function
     */
    void deserialize_px(PxIStringStream &pstream) override;

  private:
    /** @brief The XGraph */
    XGraphHolder xg_;
    /** @brief The target device */
    std::string target_;
    /** @brief The runtime to be used */
    std::string runtime_;
    /** @brief The input tensor identifiers */
    std::vector<std::string> in_tensor_names_;
    /** @brief The output tensor identifiers */
    std::vector<std::string> out_tensor_names_;
    /** @brief The run options */
    RunOptionsHolder run_options_;
    /** @brief The counter for counting the number of provided inputs */
    int count_ = 0;
    /** @brief Whether the provided target is supported on this device */
    bool is_target_supported_;
    /** @brief The internal compute function */
    ComputeFuncHolder cf_; //= nullptr;
    /** @brief The inernal quantization function */
    OpaqueFuncHolder quant_of_;
};

} // namespace runtime
} // namespace pyxir
