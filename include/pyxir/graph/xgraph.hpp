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

#include <cctype>
#include <string>
#include <iostream>
#include <exception>
#include <unordered_map>
#include <unordered_set>

#include "../common/util.hpp"
#include "xlayer.hpp"

namespace pyxir {
namespace graph {


class XGraph {

  public:

    XGraph(const std::string &name_) : name(name_) { }
    // std::cout << "Construct XGraph " << this << std::endl;

    void copy(const XGraph &xg)
    {
      name = xg.name;
      heads = xg.heads;
      tails = xg.tails;
      xlayers = xg.xlayers;
      meta_attrs = xg.meta_attrs;
    }

    // GETTERS & SETTERS //

    std::string &get_name() { return name; }
    
    void set_name(const std::string& name_) { name = name_; }

    inline bool has_meta_attr(const std::string &attr_name)
    {
      return meta_attrs.find(attr_name) != meta_attrs.end();
    }

    XAttr &get_meta_attr(const std::string &attr_name)
    { 
      if (!has_meta_attr(attr_name))
        throw std::invalid_argument("Trying to retrieve non existing meta"
                                    " attribute with name: " + attr_name
                                    + " in XGraph: " + get_name());
      return meta_attrs[attr_name];
    }

    void set_meta_attr(const std::string &attr_name, XAttr &&xattr)
    {
      meta_attrs[attr_name] = std::move(xattr);
    }
  
    std::vector<std::string> get_input_names() { return heads; }

    std::vector<std::string> get_output_names() { return tails; }

    // XLayer &get(const std::string &xl_name) { return xlayers[xl_name]; }
    std::shared_ptr<XLayer> get(const std::string &xl_name_)
    {
      std::string xl_name = pyxir::stringify(xl_name_);

      if (!contains(xl_name))
        throw std::invalid_argument(
          "Can't retrieve xlayer with name: " + xl_name
          + " as it doesn't exist.");
      return xlayers[xl_name];
    }

    inline int get_nb_inputs() { return heads.size(); }

    inline int get_nb_outputs() { return tails.size(); }

    std::vector<std::string> get_layer_names();

    int len() { return xlayers.size(); }

    // CHECKS //

    bool contains(const std::string &xl_name)
    {
      return xlayers.find(xl_name) != xlayers.end();
    }

    bool is_input(const std::string &xl_name)
    { 
      return std::find(heads.begin(), heads.end(), xl_name) != heads.end();
    }

    bool is_output(const std::string &xl_name)
    { 
      return std::find(tails.begin(), tails.end(), xl_name) != tails.end();
    }

    // GRAPH MANIPULATION //

    void add(XLayer &xl);

    void remove(const std::string &xl_name);

    void update(const std::string &xl_name);

    std::unordered_map<std::string, XAttr> meta_attrs;

    // ~XGraph() { std::cout << "Delete XGraph: " << this << std::endl; }

  private:

    void remove_head(const std::string &xl_name)
    {
      for (std::vector<std::string>::iterator it = heads.begin(); 
           it != heads.end(); ++it) {
        if (*it == xl_name) { heads.erase(it); break; }
      }
    }
    
    void remove_tail(const std::string &xl_name)
    {
      for (std::vector<std::string>::iterator it = tails.begin(); 
           it != tails.end(); ++it) {
        if (*it == xl_name) { tails.erase(it); break; }
      }
    }

    std::string name;
    std::vector<std::string> heads;
    std::vector<std::string> tails;
    std::unordered_map<std::string, std::shared_ptr<XLayer>> xlayers;
};

} // graph

typedef std::shared_ptr<pyxir::graph::XGraph> XGraphHolder;

} // pyxir