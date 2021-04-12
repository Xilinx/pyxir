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

#include <iostream>
#include <memory>

#include <catch2/catch.hpp>

#include "pyxir/pyxir.hpp"
#include "pyxir/graph/xgraph.hpp"
#include "pyxir/runtime/runtime_module.hpp"
#include "pyxir/frontend/onnx.hpp"

using namespace pyxir::runtime;

TEST_CASE("Test RuntimeModule serialization")
{
  typedef std::shared_ptr<pyxir::graph::XGraph> XGraphHolder;
  
  XGraphHolder xg = pyxir::onnx::import_onnx_model("./test.onnx");

  pyxir::RunOptionsHolder run_options(new RunOptions());
  run_options->on_the_fly_quantization = true;

  pyxir::RtModHolder rt_mod = pyxir::build_rt(
    xg, "cpu", std::vector<std::string>{"x"}, std::vector<std::string>{"z"},
    "cpu-np", run_options);

  std::ostringstream osstream;
  rt_mod->serialize(osstream);

  std::istringstream isstream(osstream.str());
  pyxir::RtModHolder rt_mod_2(new RuntimeModule());
  rt_mod_2->deserialize(isstream);

  REQUIRE(rt_mod_2->get_in_tensor_names() == std::vector<std::string>{"x"});
  REQUIRE(rt_mod_2->get_out_tensor_names() == std::vector<std::string>{"z"});
}

TEST_CASE("Test RuntimeModule IO")
{
  typedef std::shared_ptr<pyxir::graph::XGraph> XGraphHolder;
  
  XGraphHolder xg = pyxir::onnx::import_onnx_model("./test.onnx");

  pyxir::RunOptionsHolder run_options(new RunOptions());
  run_options->on_the_fly_quantization = true;

  pyxir::RtModHolder rt_mod = pyxir::build_rt(
    xg, "cpu", std::vector<std::string>{"x"}, std::vector<std::string>{"z"},
    "cpu-np", run_options);

  rt_mod->save("test.rtmod");

  pyxir::RtModHolder rt_mod_2 = RuntimeModule::Load("test.rtmod");

  REQUIRE(rt_mod_2->get_in_tensor_names() == std::vector<std::string>{"x"});
  REQUIRE(rt_mod_2->get_out_tensor_names() == std::vector<std::string>{"z"});
  pyxir::rmrf("test.rtmod");
}
