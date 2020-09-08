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
#include <sys/stat.h>
#include <fstream>

#include <catch2/catch.hpp>

#include "../util.hpp"
#include "pyxir/frontend/onnx.hpp"

// For testing private member variables
#define private public
#include "pyxir/runtime/online_quant_compute_func.hpp"


using namespace pyxir::runtime;

TEST_CASE("Test OnlineQuantComputeFunc serialization")
{
  typedef std::shared_ptr<pyxir::graph::XGraph> XGraphHolder;

  if (mkdir("test_build_dir", 0777) == -1) 
    std::cerr << "Error :  " << strerror(errno) << std::endl;

  std::string file_path = "test_build_dir/file.txt";
  std::ofstream file(file_path);
  std::string data("testtest");
  file << data;
  file.flush();
  file.close();
  
  XGraphHolder xg = pyxir::onnx::import_onnx_model("./test.onnx");

  pyxir::RunOptionsHolder run_options(new RunOptions());
  run_options->on_the_fly_quantization = true;
  run_options->build_dir = "test_build_dir";

  OnlineQuantComputeFunc cf = OnlineQuantComputeFunc(
    xg, "cpu", std::vector<std::string>{"x"}, std::vector<std::string>{"z"},
    "cpu-np", run_options);

  std::ostringstream osstream;
  cf.serialize(osstream);

  if (rmrf("test_build_dir") == -1) 
    std::cerr << "Error :  " << strerror(errno) << std::endl;

  std::istringstream isstream(osstream.str());
  OnlineQuantComputeFunc cf_2;
  cf_2.deserialize(isstream);

  REQUIRE(cf_2.target_ == "cpu");
  REQUIRE(cf_2.runtime_ == "cpu-np");
  REQUIRE(cf_2.in_tensor_names_ == std::vector<std::string>{"x"});
  REQUIRE(cf_2.out_tensor_names_ == std::vector<std::string>{"z"});
  REQUIRE(cf_2.run_options_->on_the_fly_quantization == true);
  REQUIRE(cf_2.count_ == 0);

  // CHECK BUILD_DIR
  std::ifstream in_file(file_path);
  std::string in_data;
  in_file >> in_data;
  REQUIRE(in_data == "testtest");
  in_file.close();

  // CHECK XGRAPH
  REQUIRE(cf_2.xg_->get_name() == "test-model");
  REQUIRE(cf_2.xg_->get_name() == "test-model");
  REQUIRE(cf_2.xg_->len() == 4);
  REQUIRE(cf_2.xg_->contains("x"));
  REQUIRE(cf_2.xg_->contains("y_Conv"));
  REQUIRE(cf_2.xg_->contains("y"));
  REQUIRE(cf_2.xg_->contains("z"));

  REQUIRE(cf_2.xg_->get("x")->xtype.size() == 1);
  REQUIRE(cf_2.xg_->get("x")->xtype[0] == "Input");
  REQUIRE(cf_2.xg_->get("x")->shapes.size() == 1);
  REQUIRE(cf_2.xg_->get("x")->shapes[0].size() == 4);
  REQUIRE(cf_2.xg_->get("x")->shapes[0][0] == -1);
  REQUIRE(cf_2.xg_->get("x")->get_attr("onnx_id").get_string() == "x");

  REQUIRE(cf_2.xg_->get("y_Conv")->xtype.size() == 1);
  REQUIRE(cf_2.xg_->get("y_Conv")->xtype[0] == "Convolution");
  REQUIRE(cf_2.xg_->get("y_Conv")->shapes.size() == 1);
  REQUIRE(cf_2.xg_->get("y_Conv")->shapes[0].size() == 4);
  REQUIRE(cf_2.xg_->get("y_Conv")->shapes[0][1] == 2);
  REQUIRE(cf_2.xg_->get("y_Conv")->get_attr("padding").get_ints2d().size() == 4);
  REQUIRE(cf_2.xg_->get("y_Conv")->get_attr("strides").get_ints().size() == 2);
  REQUIRE(cf_2.xg_->get("y_Conv")->get_attr("dilation").get_ints().size() == 2);
  REQUIRE(cf_2.xg_->get("y_Conv")->get_attr("kernel_size").get_ints().size() == 2);
  REQUIRE(cf_2.xg_->get("y_Conv")->get_attr("channels").get_ints().size() == 2);
  REQUIRE(cf_2.xg_->get("y_Conv")->get_attr("data_layout").get_string() == "NCHW");
  REQUIRE(cf_2.xg_->get("y_Conv")->get_attr("kernel_layout").get_string() == "OIHW");
  REQUIRE(cf_2.xg_->get("y_Conv")->get_attr("groups").get_int() == 1);
  REQUIRE(cf_2.xg_->get("y_Conv")->get_attr("onnx_id").get_string() == "y");

  REQUIRE(cf_2.xg_->get("y")->xtype.size() == 1);
  REQUIRE(cf_2.xg_->get("y")->xtype[0] == "BiasAdd");
  REQUIRE(cf_2.xg_->get("y")->shapes.size() == 1);
  REQUIRE(cf_2.xg_->get("y")->shapes[0].size() == 4);
  REQUIRE(cf_2.xg_->get("y")->shapes[0][2] == 4);
  REQUIRE(cf_2.xg_->get("y")->get_attr("onnx_id").get_string() == "y");

  REQUIRE(cf_2.xg_->get("z")->name == "z");
  REQUIRE(cf_2.xg_->get("z")->xtype.size() == 1);
  REQUIRE(cf_2.xg_->get("z")->xtype[0] == "Pooling");
  REQUIRE(cf_2.xg_->get("z")->shapes.size() == 1);
  REQUIRE(cf_2.xg_->get("z")->shapes[0].size() == 4);
  REQUIRE(cf_2.xg_->get("z")->shapes[0][3] == 2);
  REQUIRE(cf_2.xg_->get("z")->get_attr("padding").get_ints2d().size() == 4);
  REQUIRE(cf_2.xg_->get("z")->get_attr("strides").get_ints() == std::vector<int64_t>{2, 2});
  REQUIRE(cf_2.xg_->get("z")->get_attr("kernel_size").get_ints() == std::vector<int64_t>{2, 2});
  REQUIRE(cf_2.xg_->get("z")->get_attr("pool_type").get_string() == "Avg");
  REQUIRE(cf_2.xg_->get("z")->get_attr("type").get_string() == "Avg");
  REQUIRE(cf_2.xg_->get("z")->get_attr("onnx_id").get_string() == "z");

  std::ostringstream osstream2;
  cf_2.serialize(osstream2);

  if (rmrf("test_build_dir") == -1) 
    std::cerr << "Error :  " << strerror(errno) << std::endl;

  // REQUIRE(osstream2.str() == osstream.str());

  // 
}