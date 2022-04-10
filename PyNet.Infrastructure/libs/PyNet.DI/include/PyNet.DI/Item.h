#pragma once
#include <functional>
#include <typeindex>
#include <memory>
#include <any>
#include "InstanceMode.h"

using namespace std;

namespace PyNet::DI {

    struct Item
    {
        shared_ptr<void*> instancePtr;                              
        bool marker = false; 

        std::type_index type = std::type_index(typeid(void));
        function<void* (any&)> factory;
        InstanceMode instanceMode = InstanceMode::Shared;

        // non-copyable, non-moveable
        Item() = default;
        Item(const Item& rhs) = delete;
        Item& operator=(const Item& rhs) = delete;
        Item(Item&& rhs) = delete;
        Item& operator=(Item&& rhs) = delete;
    };
}