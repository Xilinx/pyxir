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

#include <unordered_map>

#include "pyxir/runtime/kernel_func_factory.hpp"


namespace pyxir {
namespace runtime {

class KernelFuncFactory::Manager
{

  private:
    Manager() {}

  public:

    typedef std::unordered_map<std::string, KernelFuncFactoryHolder> KFFMap;

    static Manager &GetInstance()
    {
      static Manager m;
      return m;
    }

    /**
     * @brief Add a kernel func factory for the given kernel id
     */
    inline void add(const std::string &kernel_id,
                    KernelFuncFactoryHolder &kff,
                    bool override = false)
    {
      if (override == false && exists(kernel_id))
        throw std::invalid_argument("KernelFuncFactory with name: " +
                                    kernel_id + " already exists.");
      kff_map_[kernel_id] = std::move(kff);
    }

    inline bool exists(const std::string &name)
    {
      return kff_map_.find(name) != kff_map_.end();
    }

    inline KernelFuncFactoryHolder &get(const std::string &name)
    {
      if (!exists(name))
        throw std::invalid_argument("KernelFuncFactory with name: " + name 
                                    + " doesn't exist.");
      return kff_map_[name];
    }

    inline void remove(const std::string &name) { kff_map_.erase(name); }

    inline const std::vector<std::string> get_names()
    {
      std::vector<std::string> names;
      for (KFFMap::iterator it = kff_map_.begin(); it != kff_map_.end(); ++it)
        names.push_back(it->first);
      return names;
    }

    inline int size() { return kff_map_.size(); }

    inline void clear() { kff_map_.clear(); }

    Manager(Manager const&) = delete;
    void operator=(Manager const&) = delete;

    // std::cout << "Delete manager " << this << std::endl; 
    ~Manager() { }

  private:
    KFFMap kff_map_;
  
};

KernelFuncFactory &
KernelFuncFactory::RegisterImpl(const std::string &kernel_id)
{
  // TODO make thread safe
  KernelFuncFactoryHolder kff(new KernelFuncFactory());
  Manager::GetInstance().add(kernel_id, kff);
  // std::cout << "Register Manager instance: " << &Manager::GetInstance() << std::endl;
  return *Manager::GetInstance().get(kernel_id);
}

bool KernelFuncFactory::Exists(const std::string &kernel_id)
{
  return Manager::GetInstance().exists(kernel_id);
}

KernelFuncHolder
KernelFuncFactory::GetKernelFunc(
  const std::string &kernel_id,
  XLayerHolder &xl)
{
  if (Manager::GetInstance().exists(kernel_id))
  {
    KernelFuncHolder kfh;
    Manager::GetInstance().get(kernel_id)->get_impl()(xl, kfh);
    return kfh;
  }  
	else
  {
    throw std::runtime_error("Kernel func: " + kernel_id + " doesn't exist.");
  }
}

} // namespace runtime
} // namespace pyxir
