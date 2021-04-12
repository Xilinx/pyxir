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


TEST_CASE("Write XGraph to string stream")
{
  typedef std::shared_ptr<pyxir::graph::XGraph> XGraphHolder;

  XGraphHolder xg = pyxir::onnx::import_onnx_model("./test.onnx");

  std::ostringstream osstream;
  pyxir::write(xg, osstream);

  std::istringstream isstream(osstream.str());
  XGraphHolder xg_rec = std::make_shared<pyxir::graph::XGraph>("");
  pyxir::read(xg_rec, isstream);

  REQUIRE(xg_rec->get_name() == "test-model");
  REQUIRE(xg_rec->len() == 4);
  REQUIRE(xg_rec->contains("x"));
  REQUIRE(xg_rec->contains("y_Conv"));
  REQUIRE(xg_rec->contains("y"));
  REQUIRE(xg_rec->contains("z"));

  REQUIRE(xg_rec->get("x")->xtype.size() == 1);
  REQUIRE(xg_rec->get("x")->xtype[0] == "Input");
  REQUIRE(xg_rec->get("x")->shapes.size() == 1);
  REQUIRE(xg_rec->get("x")->shapes[0].size() == 4);
  REQUIRE(xg_rec->get("x")->shapes[0][0] == -1);
  REQUIRE(xg_rec->get("x")->get_attr("onnx_id").get_string() == "x");

  REQUIRE(xg_rec->get("y_Conv")->xtype.size() == 1);
  REQUIRE(xg_rec->get("y_Conv")->xtype[0] == "Convolution");
  REQUIRE(xg_rec->get("y_Conv")->shapes.size() == 1);
  REQUIRE(xg_rec->get("y_Conv")->shapes[0].size() == 4);
  REQUIRE(xg_rec->get("y_Conv")->shapes[0][1] == 2);
  REQUIRE(xg_rec->get("y_Conv")->get_attr("padding").get_ints2d().size() == 4);
  REQUIRE(xg_rec->get("y_Conv")->get_attr("strides").get_ints().size() == 2);
  REQUIRE(xg_rec->get("y_Conv")->get_attr("dilation").get_ints().size() == 2);
  REQUIRE(xg_rec->get("y_Conv")->get_attr("kernel_size").get_ints().size() == 2);
  REQUIRE(xg_rec->get("y_Conv")->get_attr("channels").get_ints().size() == 2);
  REQUIRE(xg_rec->get("y_Conv")->get_attr("data_layout").get_string() == "NCHW");
  REQUIRE(xg_rec->get("y_Conv")->get_attr("kernel_layout").get_string() == "OIHW");
  REQUIRE(xg_rec->get("y_Conv")->get_attr("groups").get_int() == 1);
  REQUIRE(xg_rec->get("y_Conv")->get_attr("onnx_id").get_string() == "y");

  REQUIRE(xg_rec->get("y")->xtype.size() == 1);
  REQUIRE(xg_rec->get("y")->xtype[0] == "BiasAdd");
  REQUIRE(xg_rec->get("y")->shapes.size() == 1);
  REQUIRE(xg_rec->get("y")->shapes[0].size() == 4);
  REQUIRE(xg->get("y")->shapes[0][2] == 4);
  REQUIRE(xg_rec->get("y")->shapes[0][2] == 4);
  REQUIRE(xg_rec->get("y")->get_attr("onnx_id").get_string() == "y");

  REQUIRE(xg_rec->get("z")->name == "z");
  REQUIRE(xg_rec->get("z")->xtype.size() == 1);
  REQUIRE(xg_rec->get("z")->xtype[0] == "Pooling");
  REQUIRE(xg_rec->get("z")->shapes.size() == 1);
  REQUIRE(xg_rec->get("z")->shapes[0].size() == 4);
  REQUIRE(xg_rec->get("z")->shapes[0][3] == 2);
  REQUIRE(xg_rec->get("z")->get_attr("padding").get_ints2d().size() == 4);
  REQUIRE(xg_rec->get("z")->get_attr("strides").get_ints() == std::vector<int64_t>{2, 2});
  REQUIRE(xg_rec->get("z")->get_attr("kernel_size").get_ints() == std::vector<int64_t>{2, 2});
  REQUIRE(xg_rec->get("z")->get_attr("pool_type").get_string() == "Avg");
  REQUIRE(xg_rec->get("z")->get_attr("type").get_string() == "Avg");
  REQUIRE(xg_rec->get("z")->get_attr("onnx_id").get_string() == "z");

}
