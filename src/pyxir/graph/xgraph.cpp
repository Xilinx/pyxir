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

#include "pyxir/graph/xgraph.hpp"

#include <functional>
#include <cassert>
#include <unordered_set>

namespace pyxir {
namespace graph {

std::vector<std::string> XGraph::get_layer_names() 
{
  std::vector<std::string> layers;
  std::unordered_set<std::string> visited;
  std::function<void(std::string, std::vector<std::string> &,
                     std::unordered_set<std::string> &)> _get_rec;
  
  _get_rec = [this, &_get_rec](std::string current_,
                               std::vector<std::string> &layers_,
                               std::unordered_set<std::string> &visited_)
                              -> void
  {
    // XLayer &cX = this->get(current_);
    std::shared_ptr<XLayer> cX = this->get(current_);
    if (!cX->bottoms.empty())
      for (auto b : cX->bottoms)
        if (visited_.find(b) == visited_.end())
          _get_rec(b, layers_, visited_);
      
    layers_.push_back(current_);
    visited_.insert(current_);
  };

  for (auto tail : tails)
    _get_rec(tail, layers, visited);

  return layers;
}

void XGraph::update(const std::string &xl_name)
{
  // XLayer &xl = get(xl_name);
  std::shared_ptr<XLayer> xl = get(xl_name);
 
  // Check if bottom layers in graph
  for (auto b : xl->bottoms) {
    if (!contains(b))
      throw std::invalid_argument(
        "Invalid xlayer with name: " + xl->name + " as the bottom"
        + "layer: " + b + " doesn't exist.");
  }
  
  // Update bottom layers
  for (auto b : xl->bottoms) {

    std::shared_ptr<XLayer> bX = xlayers[b];

    if (!bX->has_top(xl->name)) {
      bX->add_top(xl->name);
      if (is_output(bX->name))
        remove_tail(bX->name);
    }
      
  }

  // Check if top layers in graph
  for (auto t : xl->tops) {
    if (!contains(t))
      throw std::invalid_argument(
        "Invalid xlayer with name: " + xl->name + " as the top"
        + "layer: " + t + " doesn't exist.");
  }

  // Update top layers
  for (auto t : xl->tops) {
    std::shared_ptr<XLayer> tX = xlayers[t];

    if (!tX->has_bottom(xl->name)) {
      tX->add_bottom(xl->name);
      if (is_input(tX->name))
        remove_head(tX->name);
    }
      
  }

  // Possibly update heads and tails
  if (xl->is_input() && !is_input(xl_name))
    heads.push_back(xl->name);

  if (xl->tops.empty() && !is_output(xl_name))
    tails.push_back(xl->name);
}

void XGraph::add(XLayer &xl)
{
  if (contains(xl.name))
    throw std::invalid_argument("Could not add xlayer with name: "
                                + xl.name + "as the layer already"
                                + " exists.");

  xlayers.insert({ xl.name, std::make_shared<XLayer>(xl) });

  update(xl.name);

  // Keep track of unique idx (equal to the position at which the layer was added)
  xidx_[xl.name] = idx_;
  xidx_re_[idx_] = xl.name;
  ++idx_;
}

void XGraph::remove(const std::string &xl_name) 
{
  std::shared_ptr<XLayer> xl = get(xl_name);

  for (auto b : xl->bottoms) {
    std::shared_ptr<XLayer> bX = get(b);
    assert(bX->has_top(xl_name));
    bX->remove_top(xl_name);

    if (bX->tops.empty())
      tails.push_back(b);
  }

  for (auto t : xl->tops) {
    std::shared_ptr<XLayer> tX = get(t);
    assert(tX->has_bottom(xl_name));
    tX->remove_bottom(xl_name);

    if (tX->bottoms.empty())
      heads.push_back(t);
  }

  xlayers.erase(xl_name);

  if (is_input(xl_name))
    remove_head(xl_name);

  if (is_output(xl_name))
    remove_tail(xl_name);

  // Remove idx
  xidx_re_.erase(xidx_[xl_name]);
  xidx_.erase(xl_name);
}

} // namespace graph
} // namespace pyxir
