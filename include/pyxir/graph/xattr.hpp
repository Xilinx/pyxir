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
#include <stdexcept>

namespace pyxir {
namespace graph {

struct XAttr {

  typedef std::unordered_map<std::string, std::vector<std::string>> MapStrVectorStr;
  typedef std::unordered_map<std::string, std::string> MapStrStr;

  std::string name;
  std::string type;
  union {
    bool b;
    int i;
    std::vector<int64_t> *ints;
    std::vector<std::vector<int64_t>> *ints2d;
    double f;
    std::vector<double> *floats;
    std::string *s;
    std::vector<std::string> *strings;
    MapStrStr *map_str_str;
    MapStrVectorStr *map_str_vstr;
  };

  XAttr() : name("empty"), type("UNDEFINED") {}
  XAttr(const XAttr& xa)
  { 
    init(xa);
  }

  XAttr(const std::string &name_) : name(name_), type("UNDEFINED") { }

  XAttr(const std::string &name_, bool b_) 
        : name(name_), type("BOOL"), b(b_) { }

  XAttr(const std::string &name_, int i_) 
        : name(name_), type("INT"), i(i_) { }

  XAttr(const std::string &name_, const std::vector<int64_t> ints_) 
        : name(name_), type("INTS") { set_ints(ints_); }

  XAttr(const std::string &name_,
        const std::vector<std::vector<int64_t>> ints2d_) 
        : name(name_), type("INTS2D") { set_ints2d(ints2d_); }

  XAttr(const std::string &name_, double f_) 
        : name(name_), type("FLOAT"), f(f_) { }

  XAttr(const std::string &name_, const std::vector<double> floats_) 
        : name(name_), type("FLOATS") { set_floats(floats_); }

  XAttr(const std::string &name_, const std::string s_) 
        : name(name_), type("STRING") { set_string(s_); }

  XAttr(const std::string &name_, const std::vector<std::string> strings_) 
        : name(name_), type("STRINGS") { set_strings(strings_); }

  XAttr(const std::string &name_, const MapStrStr &map_str_str_) 
        : name(name_), type("MAP_STR_STR") 
  {
    set_map_str_str(map_str_str_);
  }

  XAttr(const std::string &name_, const MapStrVectorStr map_str_vstr_) 
        : name(name_), type("MAP_STR_VSTR") 
  { 
    set_map_str_vstr(map_str_vstr_); 
  }

  void clean_up(const std::string &type)
  {
    if (type == "INTS")
      delete ints;
    else if (type == "INTS2D")
      delete ints2d;
    else if (type == "FLOATS")
      delete floats;
    else if (type == "STRING")
      delete s;
    else if (type == "STRINGS")
      delete strings;
    else if (type == "MAP_STR_STR")
      delete map_str_str;
    else if (type == "MAP_STR_VSTR")
      delete map_str_vstr;
  }

  XAttr& operator =(const XAttr& xa)
  {
    // Possibly we change type so we have to clean up
    clean_up(type);

    init(xa, false);
    return *this;
  }

  void init(const XAttr& xa, bool clean_up=false) 
  {
    name = xa.name;
    type = xa.type;

    if (type == "BOOL")
      b = xa.b;
    else if (type == "INT")
      i = xa.i;
    else if (type == "INTS")
      set_ints(*xa.ints, clean_up);
    else if (type == "INTS2D")
      set_ints2d(*xa.ints2d, clean_up);
    else if (type == "FLOAT")
      f = xa.f;
    else if (type == "FLOATS")
      set_floats(*xa.floats, clean_up);
    else if (type == "STRING")
      set_string(*xa.s, clean_up);
    else if (type == "STRINGS")
      set_strings(*xa.strings, clean_up);
    else if (type == "MAP_STR_STR")
      set_map_str_str(*xa.map_str_str, clean_up);
    else if (type == "MAP_STR_VSTR")
      set_map_str_vstr(*xa.map_str_vstr, clean_up);
  }

  bool get_bool()
  {
    if (type != "BOOL")
      throw std::runtime_error("Trying to retrieve XAttr value of type: "
                               + type + " as type: BOOL");
    return b;
  }

  int get_int()
  {
    if (type != "INT")
      throw std::runtime_error("Trying to retrieve XAttr value of type: "
                               + type + " as type: INT");
    return i;
  }

  std::vector<int64_t> &get_ints()
  {
    if (type != "INTS")
      throw std::runtime_error("Trying to retrieve XAttr value of type: "
                               + type + " as type: INTS");
    return *ints;
  }
  void set_ints(const std::vector<int64_t> ints_, bool clean_up=false) { 
    if (clean_up)
      delete ints;
    ints = new std::vector<int64_t>(ints_);
  }

  std::vector<std::vector<int64_t>> &get_ints2d()
  {
    if (type != "INTS2D")
      throw std::runtime_error("Trying to retrieve XAttr value of type: "
                               + type + " as type: INTS2D"); 
    return *ints2d;
  }

  void set_ints2d(const std::vector<std::vector<int64_t>> ints2d_,
                  bool clean_up=false)
  {
    if (clean_up)
      delete ints2d; 
    ints2d = new std::vector<std::vector<int64_t>>(ints2d_); 
  }

  float get_float()
  {
    if (type != "FLOAT")
      throw std::runtime_error("Trying to retrieve XAttr value of type: "
                               + type + " as type: FLOAT");
    return f;
  }

  std::vector<double> &get_floats() {
    if (type != "FLOATS")
      throw std::runtime_error("Trying to retrieve XAttr value of type: "
                               + type + " as type: FLOATS");
    return *floats; 
  }

  void set_floats(const std::vector<double> floats_, bool clean_up=false) 
  { 
    if (clean_up)
      delete floats;
    floats = new std::vector<double>(floats_); 
  }

  std::string &get_string() 
  { 
    if (type != "STRING")
      throw std::runtime_error("Trying to retrieve XAttr value of type: "
                               + type + " as type: STRING");
    return *s;
  }

  void set_string(const std::string s_, bool clean_up=false)
  {
      if (clean_up)
        delete s;
      s = new std::string(s_); 
  }

  std::vector<std::string> &get_strings()
  { 
    if (type != "STRINGS")
      throw std::runtime_error("Trying to retrieve XAttr value of type: "
                               + type + " as type: STRINGS");
    return *strings;
  }

  void set_strings(const std::vector<std::string> strings_,
                   bool clean_up=false) 
  { 
    if (clean_up)
      delete strings;
    strings = new std::vector<std::string>(strings_);
  }

  MapStrStr &get_map_str_str() 
  { 
    if (type != "MAP_STR_STR")
      throw std::runtime_error("Trying to retrieve XAttr value of type: "
                               + type + " as type: MAP_STR_STR");
    return *map_str_str;
  }

  void set_map_str_str(const MapStrStr map_str_str_,
                       bool clean_up=false) 
  { 
    if (clean_up)
      delete map_str_str;
    map_str_str = new MapStrStr(map_str_str_); 
  }

  MapStrVectorStr &get_map_str_vstr() 
  { 
    if (type != "MAP_STR_VSTR")
      throw std::runtime_error("Trying to retrieve XAttr value of type: "
                               + type + " as type: MAP_STR_VSTR");
    return *map_str_vstr;
  }
  void set_map_str_vstr(const MapStrVectorStr map_str_vstr_,
                            bool clean_up=false) 
  { 
    if (clean_up)
      delete map_str_vstr;
    map_str_vstr = new MapStrVectorStr(map_str_vstr_); 
  }

  ~XAttr() { clean_up(type); }

};

} // graph
} // pyxir