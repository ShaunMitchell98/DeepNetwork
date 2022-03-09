module;
#include <functional>
#include <typeindex>
#include <memory>
export module PyNet.DI:Item;

import :InstanceMode;

using namespace std;

namespace PyNet::DI {

    struct Item
    {
        shared_ptr<void*> instancePtr;                              
        bool marker = false; 

        //function<void*(function<typename Args::element_type>()...)> factory;
        std::type_index derivedType = std::type_index(typeid(void));    
        InstanceMode instanceMode = InstanceMode::Shared;

        // non-copyable, non-moveable
        Item() = default;
        Item(const Item& rhs) = delete;
        Item& operator=(const Item& rhs) = delete;
        Item(Item&& rhs) = delete;
        Item& operator=(Item&& rhs) = delete;
    };
}
