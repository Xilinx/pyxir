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

#include <set>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>

#include "../common/xbuffer.hpp"
#include "xattr.hpp"

namespace pyxir {
namespace graph {

struct XLayer {

  std::string name;
  std::vector<std::string> xtype;
  std::vector<std::vector<int64_t>> shapes;
  std::string shapes_t;
  std::vector<int64_t> sizes;
  std::vector<std::string> bottoms;
  std::vector<std::string> tops;
  std::vector<std::string> layer;
  std::vector<XBuffer> data;
  std::vector<std::string> targets;
  std::string target;
  std::string subgraph;
  //std::unique_ptr<XGraph> subgraph_data;
  std::vector<XLayer> *subgraph_data = nullptr;
  bool internal;
  std::unordered_map<std::string, XAttr> attrs;

  static std::set<std::string> input_types_;

  XLayer(const XLayer &xl) 
         : name(xl.name), xtype(xl.xtype), shapes(xl.shapes),
           shapes_t(xl.shapes_t), sizes(xl.sizes), bottoms(xl.bottoms),
           tops(xl.tops), layer(xl.layer), data(xl.data), targets(xl.targets),
           target(xl.target), subgraph(xl.subgraph), internal(xl.internal),
           attrs(xl.attrs)
  {
    set_subgraph_data(xl.subgraph_data);
  };

  XLayer(const std::string &name_ = std::string(),
         const std::vector<std::string> &xtype_ = std::vector<std::string>(),
         const std::vector<std::vector<int64_t>> &shapes_ = std::vector<std::vector<int64_t>>(),
         const std::string shapes_t_ = std::string("TensorShape"),
         const std::vector<int64_t> &sizes_ = std::vector<int64_t>(),
         const std::vector<std::string> &bottoms_ = std::vector<std::string>(),
         const std::vector<std::string> &tops_ = std::vector<std::string>(),
         const std::vector<std::string> &layer_ = std::vector<std::string>(),
         const std::vector<XBuffer> &data_ = std::vector<XBuffer>(),
         const std::vector<std::string> &targets_ = std::vector<std::string>(),
         const std::string &target_ = std::string(),
         const std::string &subgraph_ = std::string(),
         const bool internal_ = false,
         const std::unordered_map<std::string, XAttr> attrs_ = 
           std::unordered_map<std::string, XAttr>()
         ) 
         : name(name_), xtype(xtype_), shapes(shapes_), shapes_t(shapes_t_),
           sizes(sizes_), bottoms(bottoms_), tops(tops_), layer(layer_),
           targets(targets_), target(target_), subgraph(subgraph_),
           internal(internal_), attrs(attrs_) 
  {
    set_data(data_);
    subgraph_data = new std::vector<XLayer>();
  }

  XLayer &operator =(const XLayer &xl)
  {
    init(xl);
    return *this;
  };

  void init(const XLayer &xl)
  {
    name = xl.name;
    xtype = xl.xtype;
    shapes = xl.shapes;
    shapes_t = xl.shapes_t;
    sizes = xl.sizes;
    bottoms = xl.bottoms;
    tops = xl.tops;
    layer = xl.layer;
    data = xl.data;
    targets = xl.targets;
    target = xl.target;
    subgraph = xl.subgraph;
    internal = xl.internal;
    attrs = xl.attrs;
    set_subgraph_data(xl.subgraph_data);
  }

  const std::string &get_name() { return name; }

  std::vector<XBuffer> &get_data() { return data; }

  void set_data(const std::vector<XBuffer> &data_) {
    // TODO: 2 copies done of original buffer data before being added to XLayer 
    //  object
    data.clear();
    for (auto xb : data_)
      data.push_back(XBuffer(xb));
  }

  std::vector<XLayer> *get_subgraph_data()
  { 
    return subgraph_data;
  }

  void set_subgraph_data(const std::vector<XLayer> &subgraph_data_)
  {
    if (subgraph_data)
      delete subgraph_data;
    
    subgraph_data = new std::vector<XLayer>();
    for (auto const &e : subgraph_data_)
      subgraph_data->push_back(e);
  }

  void set_subgraph_data(std::vector<XLayer> *subgraph_data_)
  {
    if (subgraph_data)
      delete subgraph_data;

    subgraph_data = new std::vector<XLayer>();
    for (auto const e : *subgraph_data_)
      subgraph_data->push_back(e);
  }

  bool is_input() { return input_types_.find(xtype[0]) != input_types_.end(); }

  // TOPS & BOTTOMS FUNCTIONALITY

  bool has_top(const std::string &top_name)
  {
    return std::find(tops.begin(), tops.end(), top_name) != tops.end();
  }

  void add_top(const std::string &top_name)
  {
    tops.push_back(top_name);
  }

  void remove_top(const std::string &top_name) 
  {
    for (std::vector<std::string>::iterator it = tops.begin(); 
          it != tops.end(); ++it) {
      if (*it == top_name) { tops.erase(it); break; }
    }
  }

  bool has_bottom(const std::string &bottom_name)
  {
    return std::find(bottoms.begin(), bottoms.end(), bottom_name) 
      != bottoms.end();
  }

  void add_bottom(const std::string &bottom_name)
  {
    bottoms.push_back(bottom_name);
  }

  void remove_bottom(const std::string &bottom_name) 
  {
    for (std::vector<std::string>::iterator it = bottoms.begin(); 
          it != bottoms.end(); ++it) {
      if (*it == bottom_name) { bottoms.erase(it); break; }
    }
  }

  // ATTRS FUNCTIONALITY

  inline bool has_attr(const std::string &attr_name)
  {
    return attrs.find(attr_name) != attrs.end();
  }

  XAttr &get_attr(const std::string &attr_name)
  { 
    if (!has_attr(attr_name))
      throw std::invalid_argument("Trying to retrieve non existing attribute"
                                  " with name: " + attr_name + " in XLayer: "
                                  + get_name());
    return attrs[attr_name];
  }

  void set_attr(const std::string &attr_name, XAttr &&xattr)
  {
    attrs[attr_name] = std::move(xattr);
  }

  ~XLayer()
  {
    delete subgraph_data;
  }
};

} // graph

typedef std::shared_ptr<pyxir::graph::XLayer> XLayerHolder;

} // pyxir