#pragma once

#include <functional>
#include <typeindex>
#include <variant>
#include <any>

namespace PyNet::DI {

    enum class InstanceMode {
        Unique,
        Shared
    };

    struct Item
    {
        std::variant<std::monostate, std::unique_ptr<std::any>, std::shared_ptr<std::any>> instancePtr;                                   // object instance pointer
        bool marker = false;                                            // flag used to detect circular dependencies
        std::function<void(void)> factory;                              // factory fn. to create a new object instance
        std::type_index derivedType = std::type_index(typeid(void));    // a derived type (eg. implementation of an interface)
        InstanceMode instanceMode = InstanceMode::Shared;

        // non-copyable, non-moveable
        Item() = default;
        Item(const Item& rhs) = delete;
        Item& operator=(const Item& rhs) = delete;
        Item(Item&& rhs) = delete;
        Item& operator=(Item&& rhs) = delete;
    };
}
