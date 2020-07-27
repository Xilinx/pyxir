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

#include "pyxir/opaque_func_registry.hpp"

namespace pyxir {

/**
 * @brief Implementation of internal Manager singleton structure based on 
 *  https://stackoverflow.com/questions/1008019/c-singleton-design-pattern
 */
class OpaqueFuncRegistry::Manager
{

  private:
    Manager() {}

  public:

    typedef std::unordered_map<std::string, OpaqueFuncRegistryPtr> OFMap;

    static Manager &GetInstance()
    {
      static Manager m;
      return m;
    }

    inline void add(const std::string &name, OpaqueFuncRegistryPtr &of_registry)
    {
      if (exists(name))
        throw std::invalid_argument("OpaqueFunc with name: " + name 
                                    + " already exists.");
      of_map[name] = std::move(of_registry);
    }

    inline bool exists(const std::string &name)
    {
      return of_map.find(name) != of_map.end();
    }

    inline OpaqueFuncRegistryPtr get(const std::string &name)
    {
      if (!exists(name))
        throw std::invalid_argument("OpaqueFunc with name: " + name 
                                     + " doesn't exist.");
      return of_map[name];
    }

    inline void remove(const std::string &name) { of_map.erase(name); }

    inline const std::vector<std::string> get_names()
    {
      std::vector<std::string> names;
      for (OFMap::iterator it = of_map.begin(); it != of_map.end(); ++it)
        names.push_back(it->first);
      return names;
    }

    inline int size() { return of_map.size(); }

    inline void clear() { of_map.clear(); }

    Manager(Manager const&) = delete;
    void operator=(Manager const&) = delete;

    // std::cout << "Delete manager " << this << std::endl; 
    ~Manager() { }

  private:
    OFMap of_map;
  
};

OpaqueFuncRegistry::OpaqueFuncRegistryPtr
OpaqueFuncRegistry::Register(const std::string &name)
{
  // TODO make thread safe
  OpaqueFuncRegistryPtr of_registry(new OpaqueFuncRegistry());
  Manager::GetInstance().add(name, of_registry);
  // std::cout << "Register Manager instance: " << &Manager::GetInstance() << std::endl;
  return Manager::GetInstance().get(name);
}

bool OpaqueFuncRegistry::Exists(const std::string &name)
{
  return Manager::GetInstance().exists(name);
}

OpaqueFunc OpaqueFuncRegistry::Get(const std::string &name)
{
  return Manager::GetInstance().get(name)->get_func();
}

void OpaqueFuncRegistry::Remove(const std::string &name)
{
  return Manager::GetInstance().remove(name);
}

const std::vector<std::string> OpaqueFuncRegistry::GetRegisteredFuncs()
{
  // std::cout << "GetRegisteredFuncs Manager instance: " << &Manager::GetInstance() << std::endl;
  return Manager::GetInstance().get_names();
}

int OpaqueFuncRegistry::Size()
{
  return Manager::GetInstance().size();
}

void OpaqueFuncRegistry::Clear()
{
  return Manager::GetInstance().clear();
}

} // pyxir
