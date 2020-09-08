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

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <algorithm>

#include "type.hpp"
#include "common/xbuffer.hpp"
#include "graph/xgraph.hpp"
#include "ffi/str_container.hpp"

namespace pyxir {

class OpaqueFunc; 

/**
 * @brief OpaqueValue for usage in OpaqueFuncs
 */ 
struct OpaqueValue {

  pxTypeCode type_code = pxUndefined;
  union {
    // int i;
    // double f;
    std::vector<int64_t> *ints;
    std::string *s;
    std::vector<std::string> *strings;
    StrContainerHolder str_c;
    BytesContainerHolder bytes_c;
    std::shared_ptr<graph::XGraph> xg;
    std::shared_ptr<XBuffer> xb;
    std::vector<std::shared_ptr<XBuffer>> *xbuffers;
    std::shared_ptr<OpaqueFunc> of;
    void *handle;
  };

  // CONSTRUCTORS

  /// @brief Implement move constructor
  OpaqueValue(OpaqueValue &&ov_)
  { 
    move(ov_);
  }

  /// @brief Implement copy constructor
  OpaqueValue(const OpaqueValue &ov_) {
    copy(ov_);
  }

  OpaqueValue(const std::vector<int64_t> &ints_)
  { 
    set_ints(ints_); 
  }

  OpaqueValue(const std::string &s_) { set_string(s_); }

  OpaqueValue(const std::vector<std::string> &strings_)
  { 
    set_strings(strings_); 
  }

  OpaqueValue(StrContainerHolder &str_c_)
  {
    set_str_container(str_c_);
  }

  OpaqueValue(BytesContainerHolder &bytes_c_)
  {
    set_bytes_container(bytes_c_);
  }

  OpaqueValue(std::shared_ptr<graph::XGraph> &xg_)
  {
    set_xgraph(xg_);
  }

  OpaqueValue(std::shared_ptr<XBuffer> &xb_)
  { 
    set_xbuffer(xb_);
  }

  OpaqueValue(const std::vector<std::shared_ptr<XBuffer>> &xbuffers_)
  { 
    set_xbuffers(xbuffers_);
  }

  OpaqueValue(std::shared_ptr<OpaqueFunc> &of_)
  { 
    set_opaque_func(of_);
  }

  /**
   * @brief Copy function from other OpaqueValue
   */
  void copy(const OpaqueValue &ov_)
  {
    switch (ov_.type_code)
    {
      case pxVInt: set_ints(*ov_.ints); break;
      case pxStrHandle: set_string(*ov_.s); break;
      case pxVStrHandle: set_strings(*ov_.strings); break;
      case pxStrContainerHandle: set_str_container(ov_.str_c); break;
      case pxBytesContainerHandle: set_bytes_container(ov_.bytes_c); break;
      case pxXGraphHandle: set_xgraph(ov_.xg); break;
      case pxXBufferHandle: set_xbuffer(ov_.xb); break;
      case pxVXBufferHandle: set_xbuffers(*ov_.xbuffers); break;
      case pxOpaqueFuncHandle: set_opaque_func(ov_.of); break;
      default: throw std::invalid_argument("OpaqueValue of unkwown type " + 
                                           px_type_code_to_string(type_code) +
                                           " in init function");
    }
  }

  /**
   * @brief Move function from other OpaqueValue
   */
  void move(OpaqueValue &ov_)
  {
    type_code = ov_.type_code;
    switch (type_code)
    {
      case pxUndefined: break;
      case pxVInt: ints = ov_.ints; ov_.set_undefined(); break;
      case pxStrHandle: s = ov_.s; ov_.set_undefined(); break;
      case pxVStrHandle: strings = ov_.strings; ov_.set_undefined(); break;
      case pxStrContainerHandle: {
        new (&str_c) StrContainerHolder{ov_.str_c};
        ov_.set_undefined();
        break;
      }
      case pxBytesContainerHandle: {
        new (&bytes_c) BytesContainerHolder{ov_.bytes_c};
        ov_.set_undefined();
        break;
      }
      case pxXGraphHandle: {
        new (&xg) std::shared_ptr<graph::XGraph>{ov_.xg};
        ov_.set_undefined();
        break;
      }
      case pxXBufferHandle: {
        new (&xb) std::shared_ptr<XBuffer>{ov_.xb};
        ov_.set_undefined();
        break;
      }
      case pxVXBufferHandle: xbuffers = ov_.xbuffers; ov_.set_undefined(); break;
      case pxOpaqueFuncHandle: {
        new (&of) std::shared_ptr<OpaqueFunc>{ov_.of};
        ov_.set_undefined();
        break;
      }
      default: throw std::invalid_argument("OpaqueValue of unkwown type " + 
                                           px_type_code_to_string(type_code) +
                                           " in init function");
    }
  }

  // OPERATORS

  /**
   * @brief Implement assignment move operator
   */
  OpaqueValue& operator =(OpaqueValue &&ov_)
  {
    if (&ov_ == this)
      return *this;

    move(ov_);
  
    return *this;
  }

  /**
   * @brief Implement assignment copy operator
   */
  OpaqueValue& operator =(const OpaqueValue &ov_)
  {
    if (&ov_ == this)
      return *this;

    copy(ov_);
  
    return *this;
  }

  // FUNCTIONALITY

  pxTypeCode get_type_code() { return type_code; };

  const std::string get_type_code_str()
  {
    return px_type_code_to_string(type_code);
  }

  int get_type_code_int() { return (int) type_code; }

  void set_undefined()
  {
    type_code = pxUndefined;
    handle = nullptr;
  }

  // INTS
  std::vector<int64_t> &get_ints()
  { 
    if (type_code != pxVInt)
      throw std::runtime_error("Trying to retrieve OpaqueValue of type: "
                               + get_type_code_str() + " as type: " 
                               + px_type_code_to_string(pxVInt));
    return *ints;
  }

  void set_ints(const std::vector<int64_t> &ints_)
  { 
    if (type_code != pxUndefined)
      clean_up();

    type_code = pxVInt;
    ints = new std::vector<int64_t>(ints_);
  }

  // STRING
  std::string &get_string()
  { 
    if (type_code != pxStrHandle)
      throw std::runtime_error("Trying to retrieve OpaqueValue of type: "
                               + get_type_code_str() + " as type: " 
                               + px_type_code_to_string(pxStrHandle));
    return *s;
  }

  void set_string(const std::string &s_)
  { 
    if (type_code != pxUndefined)
      clean_up();

    type_code = pxStrHandle;
    s = new std::string(s_);
  }

  // STRINGS
  std::vector<std::string> &get_strings()
  { 
    if (type_code != pxVStrHandle)
      throw std::runtime_error("Trying to retrieve OpaqueValue of type: "
                               + get_type_code_str() + " as type: " 
                               + px_type_code_to_string(pxVStrHandle));
    return *strings;
  }

  void set_strings(const std::vector<std::string> &strings_)
  { 
    if (type_code != pxUndefined)
      clean_up();

    type_code = pxVStrHandle;
    strings = new std::vector<std::string>(strings_);
  }

  // STR CONTAINER
  StrContainerHolder get_str_container() {
    if (type_code != pxStrContainerHandle)
      throw std::runtime_error("Trying to retrieve OpaqueValue of type: "
                               + get_type_code_str() + " as type: " 
                               + px_type_code_to_string(pxStrContainerHandle));
    return str_c;
  }

  void set_str_container(const StrContainerHolder &str_c_)
  {
    if (type_code != pxUndefined)
      clean_up();

    type_code = pxStrContainerHandle;

    new (&str_c) StrContainerHolder{str_c_};
  }

  // BYTES CONTAINER
  BytesContainerHolder get_bytes_container() {
    if (type_code != pxBytesContainerHandle)
      throw std::runtime_error("Trying to retrieve OpaqueValue of type: "
                               + get_type_code_str() + " as type: " 
                               + px_type_code_to_string(pxBytesContainerHandle));
    return bytes_c;
  }

  void set_bytes_container(const BytesContainerHolder &bytes_c_)
  {
    if (type_code != pxUndefined)
      clean_up();

    type_code = pxBytesContainerHandle;

    new (&bytes_c) BytesContainerHolder{bytes_c_};
  }

  // XGRAPH
  std::shared_ptr<graph::XGraph> get_xgraph() {
    if (type_code != pxXGraphHandle)
      throw std::runtime_error("Trying to retrieve OpaqueValue of type: "
                               + get_type_code_str() + " as type: " 
                               + px_type_code_to_string(pxXGraphHandle));
    return xg;
  }

  void set_xgraph(const std::shared_ptr<graph::XGraph> &xg_)
  {
    if (type_code != pxUndefined)
      clean_up();

    type_code = pxXGraphHandle;

    new (&xg) std::shared_ptr<graph::XGraph>{xg_};
  }

  // XBUFFER
  std::shared_ptr<XBuffer> get_xbuffer() {
    if (type_code != pxXBufferHandle)
      throw std::runtime_error("Trying to retrieve OpaqueValue of type: "
                               + get_type_code_str() + " as type: " 
                               + px_type_code_to_string(pxXBufferHandle));
    return xb;
  }

  void set_xbuffer(const std::shared_ptr<XBuffer> &xb_)
  { 
    if (type_code != pxUndefined)
      clean_up();

    type_code = pxXBufferHandle;
    new (&xb) std::shared_ptr<XBuffer>{xb_};
  }

  // XBUFFERS
  std::vector<std::shared_ptr<XBuffer>> &get_xbuffers()
  { 
    if (type_code != pxVXBufferHandle)
      throw std::runtime_error("Trying to retrieve OpaqueValue of type: "
                               + get_type_code_str() + " as type: " 
                               + px_type_code_to_string(pxVXBufferHandle));
    return *xbuffers;
  }
  
  void set_xbuffers(const std::vector<std::shared_ptr<XBuffer>> &xbuffers_)
  { 
    if (type_code != pxVXBufferHandle)
      clean_up();

    type_code = pxVXBufferHandle;
    xbuffers = new std::vector<std::shared_ptr<XBuffer>>(xbuffers_);
  }

  // OPAQUE_FUNC
  std::shared_ptr<OpaqueFunc> get_opaque_func() {
    if (type_code != pxOpaqueFuncHandle)
      throw std::runtime_error("Trying to retrieve OpaqueValue of type: "
                               + get_type_code_str() + " as type: " 
                               + px_type_code_to_string(pxOpaqueFuncHandle));
    return of;
  }

  void set_opaque_func(const std::shared_ptr<OpaqueFunc> &of_)
  { 
    if (type_code != pxUndefined)
      clean_up();

    type_code = pxOpaqueFuncHandle;
    new (&of) std::shared_ptr<OpaqueFunc>{of_};
  }


  /**
   * @brief Clean up the memory
   */
  void clean_up()
  {
    // std::cout << "clean up: " << this << ", type code: " << get_type_code_str() << std::endl;
    switch (type_code)
    {
      case pxVInt: delete ints; break;
      case pxStrHandle: delete s; break;
      case pxVStrHandle: delete strings; break;
      case pxStrContainerHandle: str_c.~shared_ptr(); break;
      case pxBytesContainerHandle: bytes_c.~shared_ptr(); break;
      case pxXGraphHandle: xg.~shared_ptr(); break;
      case pxXBufferHandle: xb.~shared_ptr(); break;
      case pxVXBufferHandle: delete xbuffers; break;
      case pxOpaqueFuncHandle: of.~shared_ptr(); break;
      default: break;
    }
  }

  ~OpaqueValue() { clean_up(); }
};


/**
 * @brief The OpaqueFunc arguments
 */
class OpaqueArgs {
  
  public:
    std::vector<std::shared_ptr<OpaqueValue>> args;
    
    OpaqueArgs() {}
    OpaqueArgs(std::vector<std::shared_ptr<OpaqueValue>> &args_) 
    {
      // std::cout << "-- Constructor Move args" << this << std::endl;
      args = std::move(args_);
    }

    OpaqueArgs(const std::vector<OpaqueValue> &args_) 
    {
      // std::cout << "-- Constructor Copy args" << this << std::endl;
      for (auto &arg_ : args_)
        args.push_back(std::make_shared<OpaqueValue>(arg_));
    }

    /**
     * @brief Return the size of the arguments
     */
    inline int size() const { return args.size(); }

    /**
     * @brief Get the i-th opaque value element
     * @param i the index
     * @return the i-th opaque value element
     */
    inline std::shared_ptr<OpaqueValue> operator[](int i) { return args[i]; }
    
    ~OpaqueArgs() { }
      // std::cout << "Delete OpaqueArgs: " << this << std::endl;
};


/**
 * @brief The opaque function is a generic function definition for FFI (Foreign
 *  function interface).
 */
class OpaqueFunc {
  
  public:

    /** @brief The type of the internal std::function */
    typedef std::function<void (OpaqueArgs &)> FuncType;

    OpaqueFunc() {
      // std::cout << "Create OpaqueFunc: " << this << std::endl;
    }

    OpaqueFunc(const FuncType &func_,
               const std::vector<pxTypeCode> &arg_type_codes_)
    {
      // : func(func_)
      // std::cout << "Create OpaqueFunc from func: " << this << std::endl;
      //func = new FuncType(func_);
      set_func(func_, arg_type_codes_);
    }

    OpaqueFunc(const FuncType &func_,
               const std::vector<int64_t> &arg_type_codes_)
    {
      set_func(func_, arg_type_codes_);
    }

    inline void set_func(const FuncType &func_,
                         const std::vector<pxTypeCode> &arg_type_codes_)
    {
      func = func_;
      arg_type_codes = arg_type_codes_;
    }

    inline void set_func(const FuncType &func_,
                         const std::vector<int64_t> &arg_type_codes_)
    { 
      func = func_;
      arg_type_codes.clear();

      std::transform(arg_type_codes_.begin(), arg_type_codes_.end(),
        std::back_inserter(arg_type_codes), [](int i) { 
          return static_cast<pxTypeCode>(i);
        });
    }

    inline FuncType get_func() { return func; }

    inline const std::vector<int64_t> get_arg_type_codes()
    {
      std::vector<int64_t> int_arg_type_codes;
      std::transform(arg_type_codes.begin(), arg_type_codes.end(),
        std::back_inserter(int_arg_type_codes), [](pxTypeCode i) { 
          return static_cast<int>(i);
        });
      return int_arg_type_codes;
    }
    
    /**
     * @brief Call OpaqueFunc with no arguments
     */
    // inline void operator()() { call(); }
    
    /**
     * @brief Call OpaqueFunc with provided arguments
     * @param args The provided arguments as a parameter pack
     */
    template<typename... Args>
    inline void operator()(Args& ...args_)
    {
      std::vector<OpaqueValue> args_v; // = std::move(args_...); //{args_...};
      call_util(args_v, args_...);
      OpaqueArgs opaque_args = OpaqueArgs(args_v);
      call(opaque_args);
    }

    inline void call(OpaqueArgs &args_)
    { 
      // std::cout << "before func call" << std::endl;
      // std::cout << "func is " << (func ? "callable" : "not callable") << std::endl;
      func(args_);
      // std::cout << "after func call" << std::endl;
    }

    // inline void call()
    // {
    //   OpaqueArgs opaque_args = OpaqueArgs();
    //   func(opaque_args);
    // }

    ~OpaqueFunc() { }
      // std::cout << "Delete OpaqueFunc: " << this << std::endl;

  private:

    template<typename T>
    inline void call_util(std::vector<OpaqueValue> &ov_v, T &t)
    {
      ov_v.push_back(OpaqueValue(t));
    }

    template<typename T, typename... Args>
    inline void call_util(std::vector<OpaqueValue> &ov_v, T &t, Args& ...args_)
    {
      ov_v.push_back(OpaqueValue(t));
      call_util(ov_v, args_...);
    }

    /** @brief The internal std::function */
    FuncType func;
    std::vector<pxTypeCode> arg_type_codes;
};

typedef std::shared_ptr<OpaqueFunc> OpaqueFuncHolder;

} // pyxir