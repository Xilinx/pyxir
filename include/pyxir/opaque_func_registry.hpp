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
#include <memory>

#include "pyxir_api.hpp"
#include "opaque_func.hpp"

namespace pyxir {

class OpaqueFuncRegistry {
  
  public:

    typedef std::shared_ptr<OpaqueFuncRegistry> OpaqueFuncRegistryPtr;
    typedef std::shared_ptr<OpaqueFunc> OpaqueFuncHolder;
    typedef std::function<void (OpaqueArgs &)> FuncType;

    OpaqueFuncRegistry() {}

    /**
     * @brief Set the internal opaque func
     * @param func_ The opaque function to be set for this registry
     * @return Reference to tis registry
     */
    PX_API OpaqueFuncRegistry &set_func(OpaqueFunc &func_)
    {
      // std::cout << "set func func_ is " << (func_.get_func() ? "callable" : "not callable") << std::endl;
      func = func_;
      // std::cout << "set func func is " << (func.get_func() ? "callable" : "not callable") << std::endl;
      // std::cout << "registry: " << this << std::endl;
      return *this;
    }

    /**
     * @brief Initialize the internal opaque func with the given std::function
     * @param func_ The std::function to be used for the internal OpaqueFunc
     * @param args_type_codes_ The type codes of the function arguments
     * @return Reference to this registry
     */
    PX_API OpaqueFuncRegistry &set_func(FuncType &func_, 
                                        const std::vector<int64_t> &args_type_codes_)
    {
      func = OpaqueFunc(func_, args_type_codes_);
      return *this;
    }

    /**
     * @brief Initialize the internal opaque func with the given std::function
     * @param func_ The std::function to be used for the internal OpaqueFunc
     * @param args_type_codes_ The type codes of the function arguments
     * @return Reference to this registry
     */
    PX_API OpaqueFuncRegistry &set_func(FuncType &func_, 
                                        const std::vector<pxTypeCode> &args_type_codes_)
    {
      func = OpaqueFunc(func_, args_type_codes_);
      return *this;
    }

    /**
     * @brief Initialize the internal opaque func with a lambda function
     * @param func_ The lambda function to be used for the internal OpaqueFunc
     * @param args_type_codes_ The type codes of the function arguments
     * @return Reference to this registry
     */
    PX_API OpaqueFuncRegistry &set_func(void* func_, 
                                        const std::vector<int64_t> &args_type_codes_)
    {
      FuncType f = *static_cast<FuncType*>(func_);
      func = OpaqueFunc(f, args_type_codes_);
      return *this;
    }

    /**
     * @brief Initialize the internal opaque func with a lambda function
     * @param func_ The lambda function to be used for the internal OpaqueFunc
     * @param args_type_codes_ The type codes of the function arguments
     * @return Reference to this registry
     */
    PX_API OpaqueFuncRegistry &set_func(void* func_, 
                                        const std::vector<pxTypeCode> &args_type_codes_)
    {
      FuncType f = *static_cast<FuncType*>(func_);
      func = OpaqueFunc(f, args_type_codes_);
      return *this;
    }

    /**
     * @brief Initialize the internal opaque func with the given lambda
     * @param func_ The lambda to be used for the internal OpaqueFunc
     * @param args_type_codes_ The type codes of the function arguments
     * @return Reference to this registry
     */
    // template<typename F>
    PX_API OpaqueFuncRegistry &set_func(void (*func_)(OpaqueArgs &),
                                        const std::vector<int64_t> &args_type_codes_) // 
    {
      func = OpaqueFunc((FuncType) func_, args_type_codes_);
      return *this;
    }

    /**
     * @brief Initialize the internal opaque func with the given lambda
     * @param func_ The lambda to be used for the internal OpaqueFunc
     * @param args_type_codes_ The type codes of the function arguments
     * @return Reference to this registry
     */
    // template<typename F>
    PX_API OpaqueFuncRegistry &set_func(void (*func_)(OpaqueArgs &),
                                        const std::vector<pxTypeCode> &args_type_codes_) // 
    {
      func = OpaqueFunc((FuncType) func_, args_type_codes_);
      return *this;
    }

    /**
     * @brief Get the internal opaque func
     * @return Reference to internal opaque func
     */
    PX_API OpaqueFunc get_func() { return func; }

    /**
     * @brief Get the internal opaque func argument type codes
     * @return The opaque func arguments type codes
     */
    // PX_API std::vector<int64_t> get_args_type_codes() { return args_type_codes; }

    /**
     * @brief Register a global opaque function
     * @param name The name of the opaque function
     * @return Reference to the registry
     */
    PX_API static OpaqueFuncRegistryPtr Register(const std::string &name);

    /**
     * @brief Return whether an OpaqueFunc with the given name exists
     */
    PX_API static bool Exists(const std::string &name);

    /**
     * @brief Retrieve a registered opaque function
     * @param name The name of the opaque function to be retrieved
     * @return Reference to the opaque function
     */
    PX_API static OpaqueFunc Get(const std::string &name);

    /**
     * @brief Remove a registered opaque function
     * @param name The name of the opaque function to be removed
     */
    PX_API static void Remove(const std::string &name);

    /**
     * @brief Retrieve a registered opaque function names
     * @return The opaque function names
     */
    PX_API static const std::vector<std::string> GetRegisteredFuncs();

    /**
     * @brief Return the number of registered opaque functions
     */
    PX_API static int Size();

    /**
     * @brief Reset the opaque func registry
     */
    PX_API static void Clear();

    // Internal singleton Manager
    class Manager;

    ~OpaqueFuncRegistry() { }
    //std::cout << "Delete OpaqueFuncRegistry: " << this << std::endl; }

  private:
    OpaqueFunc func;
};

#define STR_CONCAT_(__x, __y) __x##__y
#define STR_CONCAT(__x, __y) STR_CONCAT_(__x, __y)

#define OPAQUE_FUNC_REG_VAR_DEF                         \
  static ::pyxir::OpaqueFuncRegistry&  __mk_ ## PX

#define REGISTER_OPAQUE_FUNC(OFName)\
  STR_CONCAT(OPAQUE_FUNC_REG_VAR_DEF, __COUNTER__) = \
  OpaqueFuncRegistry::Register(OFName)

} // pyxir
