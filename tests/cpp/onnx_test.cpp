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


TEST_CASE("Load model from ONNX file and partition")
{
  typedef std::shared_ptr<pyxir::graph::XGraph> XGraphHolder;
  
  XGraphHolder xg = pyxir::onnx::import_onnx_model("./test.onnx");
  REQUIRE(xg->get_name() == "test-model");
  REQUIRE(xg->len() == 4);
  REQUIRE(xg->contains("x"));
  REQUIRE(xg->contains("y_Conv"));
  REQUIRE(xg->contains("y"));
  REQUIRE(xg->contains("z"));

  REQUIRE(xg->get("x")->xtype.size() == 1);
  REQUIRE(xg->get("x")->xtype[0] == "Input");
  REQUIRE(xg->get("x")->shapes.size() == 1);
  REQUIRE(xg->get("x")->shapes[0].size() == 4);
  REQUIRE(xg->get("x")->shapes[0][0] == -1);
  REQUIRE(xg->get("x")->get_attr("onnx_id").get_string() == "x");

  REQUIRE(xg->get("y_Conv")->xtype.size() == 1);
  REQUIRE(xg->get("y_Conv")->xtype[0] == "Convolution");
  REQUIRE(xg->get("y_Conv")->shapes.size() == 1);
  REQUIRE(xg->get("y_Conv")->shapes[0].size() == 4);
  REQUIRE(xg->get("y_Conv")->shapes[0][1] == 2);
  REQUIRE(xg->get("y_Conv")->get_attr("padding").get_ints2d().size() == 4);
  REQUIRE(xg->get("y_Conv")->get_attr("strides").get_ints().size() == 2);
  REQUIRE(xg->get("y_Conv")->get_attr("dilation").get_ints().size() == 2);
  REQUIRE(xg->get("y_Conv")->get_attr("kernel_size").get_ints().size() == 2);
  REQUIRE(xg->get("y_Conv")->get_attr("channels").get_ints().size() == 2);
  REQUIRE(xg->get("y_Conv")->get_attr("data_layout").get_string() == "NCHW");
  REQUIRE(xg->get("y_Conv")->get_attr("kernel_layout").get_string() == "OIHW");
  REQUIRE(xg->get("y_Conv")->get_attr("groups").get_int() == 1);
  REQUIRE(xg->get("y_Conv")->get_attr("onnx_id").get_string() == "y");

  REQUIRE(xg->get("y")->xtype.size() == 1);
  REQUIRE(xg->get("y")->xtype[0] == "BiasAdd");
  REQUIRE(xg->get("y")->shapes.size() == 1);
  REQUIRE(xg->get("y")->shapes[0].size() == 4);
  REQUIRE(xg->get("y")->shapes[0][2] == 4);
  REQUIRE(xg->get("y")->get_attr("onnx_id").get_string() == "y");

  REQUIRE(xg->get("z")->name == "z");
  REQUIRE(xg->get("z")->xtype.size() == 1);
  REQUIRE(xg->get("z")->xtype[0] == "Pooling");
  REQUIRE(xg->get("z")->shapes.size() == 1);
  REQUIRE(xg->get("z")->shapes[0].size() == 4);
  REQUIRE(xg->get("z")->shapes[0][3] == 2);
  REQUIRE(xg->get("z")->get_attr("padding").get_ints2d().size() == 4);
  REQUIRE(xg->get("z")->get_attr("strides").get_ints() == std::vector<int64_t>{2, 2});
  REQUIRE(xg->get("z")->get_attr("kernel_size").get_ints() == std::vector<int64_t>{2, 2});
  REQUIRE(xg->get("z")->get_attr("pool_type").get_string() == "Avg");
  REQUIRE(xg->get("z")->get_attr("type").get_string() == "Avg");
  REQUIRE(xg->get("z")->get_attr("onnx_id").get_string() == "z");

  pyxir::partition(xg, std::vector<std::string>{"dpuv1"}, "");
  REQUIRE(xg->get_name() == "test-model");
  REQUIRE(xg->len() == 4);
  REQUIRE(xg->contains("x"));
  REQUIRE(xg->contains("y_Conv"));
  REQUIRE(xg->contains("y"));
  REQUIRE(xg->contains("z"));

  REQUIRE(xg->get("x")->xtype.size() == 1);
  REQUIRE(xg->get("x")->xtype[0] == "Input");
  REQUIRE(xg->get("x")->target == "cpu");
  REQUIRE(xg->get("x")->subgraph == "");

  REQUIRE(xg->get("y_Conv")->xtype.size() == 1);
  REQUIRE(xg->get("y_Conv")->xtype[0] == "Convolution");
  REQUIRE(xg->get("y_Conv")->target == "dpuv1");
  REQUIRE(xg->get("y_Conv")->subgraph == "xp0");

  REQUIRE(xg->get("y")->xtype.size() == 1);
  REQUIRE(xg->get("y")->xtype[0] == "BiasAdd");
  REQUIRE(xg->get("y")->target == "dpuv1");
  REQUIRE(xg->get("y")->subgraph == "xp0");

  REQUIRE(xg->get("z")->name == "z");
  REQUIRE(xg->get("z")->xtype.size() == 1);
  REQUIRE(xg->get("z")->xtype[0] == "Pooling");
  REQUIRE(xg->get("z")->target == "dpuv1");
  REQUIRE(xg->get("z")->subgraph == "xp0");

  // const std::vector<std::string> supported_tensors = 
  //   pyxir::get_supported_tensors(xg, "dpuv1");
}

TEST_CASE("Load model from ONNX file and build runtime")
{
  typedef std::shared_ptr<pyxir::graph::XGraph> XGraphHolder;
  
  XGraphHolder xg = pyxir::onnx::import_onnx_model("./test.onnx");
  REQUIRE(xg->get_name() == "test-model");
  REQUIRE(xg->len() == 4);
  REQUIRE(xg->contains("x"));
  REQUIRE(xg->contains("y_Conv"));
  REQUIRE(xg->contains("y"));
  REQUIRE(xg->contains("z"));

  pyxir::RtModHolder rt_mod = pyxir::build_rt(
    xg, "cpu", std::vector<std::string>{"x"}, std::vector<std::string>{"z"});

  std::array<float, 16> in_arr = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  pyxir::XBufferHolder xb_in = std::shared_ptr<pyxir::XBuffer>(
    new pyxir::XBuffer((void *) &in_arr[0], 4, "f", 4,
                       std::vector<ssize_t>{1, 1, 4, 4}, false, false));

  std::array<float, 8> out_arr = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  pyxir::XBufferHolder xb_out = std::shared_ptr<pyxir::XBuffer>(
    new pyxir::XBuffer((void *) &out_arr[0], 4, "f", 4,
                       std::vector<ssize_t>{1, 2, 2, 2}, false, false));
    
  // for (int i = 0; i < 8; ++i)
  //   std::cout << out_arr[i] << ", " << std::endl;

  std::vector<pyxir::XBufferHolder> in_tensors{xb_in};
  std::vector<pyxir::XBufferHolder> out_tensors{xb_out};

  rt_mod->execute(in_tensors, out_tensors);

  std::array<float, 8> expected_out = {5.0, 4.0, 4.0, 3.25, 1.0, 1.0, 0.0, 0.25};
  REQUIRE(out_tensors.size() == 1);
  REQUIRE(out_arr == expected_out);

  pyxir::RtModHolder rt_mod_np = pyxir::build_rt(
    xg, "cpu", std::vector<std::string>{"x"}, std::vector<std::string>{"z"},
    "cpu-np");

  std::array<float, 8> out_arr2 = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  pyxir::XBufferHolder xb_out2 = std::shared_ptr<pyxir::XBuffer>(
    new pyxir::XBuffer((void *) &out_arr2[0], 4, "f", 4,
                       std::vector<ssize_t>{1, 2, 2, 2}, false, false));
  std::vector<pyxir::XBufferHolder> out_tensors2{xb_out2};

  rt_mod->execute(in_tensors, out_tensors2);

  REQUIRE(out_tensors2.size() == 1);
  REQUIRE(out_arr2 == expected_out);

}

TEST_CASE("Load model from ONNX file and build online quantization runtime")
{
  typedef std::shared_ptr<pyxir::graph::XGraph> XGraphHolder;

  XGraphHolder xg = pyxir::onnx::import_onnx_model("./test.onnx");

  pyxir::RunOptionsHolder run_options(new pyxir::runtime::RunOptions());
  run_options->on_the_fly_quantization = true;

  pyxir::RtModHolder rt_mod = pyxir::build_rt(
    xg, "cpu", std::vector<std::string>{"x"}, std::vector<std::string>{"z"},
    "cpu-tf", run_options);

  std::array<float, 16> in_arr = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  pyxir::XBufferHolder xb_in = std::shared_ptr<pyxir::XBuffer>(
    new pyxir::XBuffer((void *) &in_arr[0], 4, "f", 4,
                       std::vector<ssize_t>{1, 1, 4, 4}, false, false));

  std::array<float, 8> out_arr = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  pyxir::XBufferHolder xb_out = std::shared_ptr<pyxir::XBuffer>(
    new pyxir::XBuffer((void *) &out_arr[0], 4, "f", 4,
                       std::vector<ssize_t>{1, 2, 2, 2}, false, false));
    
  // for (int i = 0; i < 8; ++i)
  //   std::cout << out_arr[i] << ", " << std::endl;

  std::vector<pyxir::XBufferHolder> in_tensors{xb_in};
  std::vector<pyxir::XBufferHolder> out_tensors{xb_out};

  rt_mod->execute(in_tensors, out_tensors);

  std::array<float, 8> expected_out = {5.0, 4.0, 4.0, 3.25, 1.0, 1.0, 0.0, 0.25};
  REQUIRE(out_tensors.size() == 1);
  REQUIRE(out_arr == expected_out);

  pyxir::RtModHolder rt_mod_np = pyxir::build_rt(
    xg, "cpu", std::vector<std::string>{"x"}, std::vector<std::string>{"z"},
    "cpu-np", run_options);

  std::array<float, 8> out_arr2 = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  pyxir::XBufferHolder xb_out2 = std::shared_ptr<pyxir::XBuffer>(
    new pyxir::XBuffer((void *) &out_arr2[0], 4, "f", 4,
                       std::vector<ssize_t>{1, 2, 2, 2}, false, false));
  std::vector<pyxir::XBufferHolder> out_tensors2{xb_out2};

  rt_mod->execute(in_tensors, out_tensors2);

  REQUIRE(out_tensors2.size() == 1);
  REQUIRE(out_arr2 == expected_out);

}
