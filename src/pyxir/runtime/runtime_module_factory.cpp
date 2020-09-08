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


#include "pyxir/runtime/runtime_module_factory.hpp"


namespace pyxir {
namespace runtime {

class RuntimeModuleFactory::Manager
{

  private:
    Manager() {}

  public:

    typedef std::unordered_map<std::string, RuntimeModuleFactoryHolder> RMFMap;

    static Manager &GetInstance()
    {
      static Manager m;
      return m;
    }

    /**
     * @brief Add a runtime module factor for the given runtime
     */
    inline void add(const std::string &runtime,
                    RuntimeModuleFactoryHolder &rmf,
                    bool override = true)
    {
      if (override == false && exists(runtime))
        throw std::invalid_argument("RuntimeModuleFactory with name: " +
                                    runtime + " already exists.");
      rmf_map_[runtime] = std::move(rmf);
    }

    inline bool exists(const std::string &name)
    {
      return rmf_map_.find(name) != rmf_map_.end();
    }

    inline RuntimeModuleFactoryHolder &get(const std::string &name)
    {
      if (!exists(name))
        throw std::invalid_argument("RuntimeModuleFactory with name: " + name 
                                    + " doesn't exist.");
      return rmf_map_[name];
    }

    inline void remove(const std::string &name) { rmf_map_.erase(name); }

    inline const std::vector<std::string> get_names()
    {
      std::vector<std::string> names;
      for (RMFMap::iterator it = rmf_map_.begin(); it != rmf_map_.end(); ++it)
        names.push_back(it->first);
      return names;
    }

    inline int size() { return rmf_map_.size(); }

    inline void clear() { rmf_map_.clear(); }

    Manager(Manager const&) = delete;
    void operator=(Manager const&) = delete;

    // std::cout << "Delete manager " << this << std::endl; 
    ~Manager() { }

  private:
    RMFMap rmf_map_;
  
};

RuntimeModuleFactory &
RuntimeModuleFactory::RegisterImpl(const std::string &runtime)
{
  // TODO make thread safe
  RuntimeModuleFactoryHolder rmf(new RuntimeModuleFactory(runtime));
  Manager::GetInstance().add(runtime, rmf);
  // std::cout << "Register Manager instance: " << &Manager::GetInstance() << std::endl;
  return *Manager::GetInstance().get(runtime);
}

RtModHolder 
RuntimeModuleFactory::GetRuntimeModule(
  std::shared_ptr<graph::XGraph> &xg,
  const std::string &target,
  const std::vector<std::string> &in_tensor_names,
  const std::vector<std::string> &out_tensor_names,
  const std::string &runtime,
  RunOptionsHolder const &run_options)
{
  return Manager::GetInstance().get(runtime)->get_impl()
    ->get_runtime_module(
      xg, target, in_tensor_names, out_tensor_names, run_options
    );
}

bool RuntimeModuleFactory::Exists(const std::string &runtime)
{
  return Manager::GetInstance().exists(runtime);
}

bool RuntimeModuleFactory::SupportsTarget(const std::string &runtime, const std::string &target)
{
  return Manager::GetInstance().get(runtime)->is_target_supported(target);
}

} // namespace runtime
} // namespace pyxir