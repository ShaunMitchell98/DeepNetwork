#pragma once

#include <functional>
#include <typeindex>
#include <any>

using namespace std;

namespace PyNet::DI {

    class Context;

    enum class InstanceMode { 
        Unique,
        Shared
    };

    struct Item
    {
        shared_ptr<void*> instancePtr;                              
        bool marker = false; 

        function<void*(Context&)> factory;                              
        type_index derivedType = type_index(typeid(void));    
        InstanceMode instanceMode = InstanceMode::Shared;

        // non-copyable, non-moveable
        Item() = default;
        Item(const Item& rhs) = delete;
        Item& operator=(const Item& rhs) = delete;
        Item(Item&& rhs) = delete;
        Item& operator=(Item&& rhs) = delete;
    };
}
