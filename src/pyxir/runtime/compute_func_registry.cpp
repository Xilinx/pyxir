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

#include "pyxir/runtime/compute_func_registry.hpp"


namespace pyxir {
namespace runtime {

class ComputeFuncRegistry::Manager
{

  private:
    Manager() {}

  public:

    typedef std::unordered_map<std::string, ComputeFuncRegistryHolder> CFRMap;

    static Manager &GetInstance()
    {
      static Manager m;
      return m;
    }

    /**
     * @brief Add a compute function factory
     */
    inline void add(const std::string &cf_type, ComputeFuncRegistryHolder &cfr)
    {
      if (exists(cf_type))
        throw std::invalid_argument("ComputeFuncRegistry with name: " +
                                    cf_type + " already exists.");
      cfr_map_[cf_type] = std::move(cfr);
    }

    inline bool exists(const std::string &cf_type)
    {
      return cfr_map_.find(cf_type) != cfr_map_.end();
    }

    inline ComputeFuncRegistryHolder &get(const std::string &cf_type)
    {
      if (!exists(cf_type))
        throw std::invalid_argument("ComputeFuncRegistry with name: " + cf_type 
                                    + " doesn't exist.");
      return cfr_map_[cf_type];
    }

    inline void remove(const std::string &cf_type) { cfr_map_.erase(cf_type); }

    inline const std::vector<std::string> get_types()
    {
      std::vector<std::string> types;
      for (CFRMap::iterator it = cfr_map_.begin(); it != cfr_map_.end(); ++it)
        types.push_back(it->first);
      return types;
    }

    inline int size() { return cfr_map_.size(); }

    inline void clear() { cfr_map_.clear(); }

    Manager(Manager const&) = delete;
    void operator=(Manager const&) = delete;

    ~Manager() { }

  private:
    CFRMap cfr_map_;
  
};

ComputeFuncRegistry &ComputeFuncRegistry::Register(const std::string &cf_type)
{
  // TODO make thread safe
  ComputeFuncRegistryHolder cff(new ComputeFuncRegistry());
  Manager::GetInstance().add(cf_type, cff);
  return *Manager::GetInstance().get(cf_type);
}

bool ComputeFuncRegistry::Exists(const std::string &cf_type)
{
  return Manager::GetInstance().exists(cf_type);
}

ComputeFuncHolder ComputeFuncRegistry::GetComputeFunc(const std::string &cf_type)
{
  return Manager::GetInstance().get(cf_type)->get_factory_func()();
}

} // namespace runtime
} // namespace pyxir
