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

#include "pyxir/runtime/compute_func_factory.hpp"


namespace pyxir {
namespace runtime {

class ComputeFuncFactory::Manager
{

  private:
    Manager() {}

  public:

    typedef std::unordered_map<std::string, ComputeFuncFactoryHolder> CFFMap;

    static Manager &GetInstance()
    {
      static Manager m;
      return m;
    }

    /**
     * @brief Add a runtime module factor for the given runtime
     */
    inline void add(const std::string &runtime,
                    ComputeFuncFactoryHolder &cff,
                    bool override = true)
    {
      if (override == false && exists(runtime))
        throw std::invalid_argument("ComputeFuncFactory with name: " +
                                    runtime + " already exists.");
      cff_map_[runtime] = std::move(cff);
    }

    inline bool exists(const std::string &name)
    {
      return cff_map_.find(name) != cff_map_.end();
    }

    inline ComputeFuncFactoryHolder &get(const std::string &name)
    {
      if (!exists(name))
        throw std::invalid_argument("ComputeFuncFactory with name: " + name 
                                    + " doesn't exist.");
      return cff_map_[name];
    }

    inline void remove(const std::string &name) { cff_map_.erase(name); }

    inline const std::vector<std::string> get_names()
    {
      std::vector<std::string> names;
      for (CFFMap::iterator it = cff_map_.begin(); it != cff_map_.end(); ++it)
        names.push_back(it->first);
      return names;
    }

    inline int size() { return cff_map_.size(); }

    inline void clear() { cff_map_.clear(); }

    Manager(Manager const&) = delete;
    void operator=(Manager const&) = delete;

    // std::cout << "Delete manager " << this << std::endl; 
    ~Manager() { }

  private:
    CFFMap cff_map_;
  
};

ComputeFuncFactory &
ComputeFuncFactory::RegisterImpl(const std::string &runtime)
{
  // TODO make thread safe
  ComputeFuncFactoryHolder cff(new ComputeFuncFactory());
  Manager::GetInstance().add(runtime, cff);
  // std::cout << "Register Manager instance: " << &Manager::GetInstance() << std::endl;
  return *Manager::GetInstance().get(runtime);
}

bool ComputeFuncFactory::Exists(const std::string &runtime)
{
  return Manager::GetInstance().exists(runtime);
}

ComputeFuncHolder
ComputeFuncFactory::GetComputeFunc(
  std::shared_ptr<graph::XGraph> &xg,
  const std::string &target,
  const std::vector<std::string> &in_tensor_names,
  const std::vector<std::string> &out_tensor_names,
  const std::string &runtime,
  RunOptionsHolder const &run_options)
{
  if (Manager::GetInstance().exists(runtime)) {
    return Manager::GetInstance().get(runtime)->get_impl()
      ->get_compute_func(
        xg, target, in_tensor_names, out_tensor_names, run_options
      );
  } else {
    return ComputeFuncFactoryImpl(runtime).get_compute_func(
      xg, target, in_tensor_names, out_tensor_names, run_options
    );
  }
}

} // namespace runtime
} // namespace pyxir
