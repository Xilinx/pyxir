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

#include "pyxir/runtime/run_options.hpp"

using namespace pyxir::runtime;


TEST_CASE("Test RunOptions serialization")
{
  std::ostringstream sstream;

  RunOptions run_options;
  run_options.build_dir = "test_build_dir";
  run_options.serialize(sstream);

  std::istringstream isstream(sstream.str());
  RunOptions run_options2;
  run_options2.deserialize(isstream);
  REQUIRE(run_options2.on_the_fly_quantization == run_options.on_the_fly_quantization);
  REQUIRE(run_options2.nb_quant_inputs == run_options.nb_quant_inputs);
  REQUIRE(run_options2.build_dir == "test_build_dir");
  REQUIRE(run_options2.work_dir == "/tmp/vitis_ai_work");
  REQUIRE(!run_options2.is_prebuilt);
}

TEST_CASE("Test RunOptions loadFromSStream")
{
  std::ostringstream sstream;

  RunOptions run_options;
  run_options.build_dir = "test_build_dir";
  run_options.serialize(sstream);

  std::istringstream isstream(sstream.str());
  RunOptions run_options2 = RunOptions::loadFromSStream<RunOptions>(isstream);
  REQUIRE(run_options2.on_the_fly_quantization == run_options.on_the_fly_quantization);
  REQUIRE(run_options2.nb_quant_inputs == run_options.nb_quant_inputs);
  REQUIRE(run_options2.build_dir == "test_build_dir");
  REQUIRE(run_options2.work_dir == "/tmp/vitis_ai_work");
  REQUIRE(!run_options2.is_prebuilt);
}
