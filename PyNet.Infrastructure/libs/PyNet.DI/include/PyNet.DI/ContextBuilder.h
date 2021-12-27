#pragma once

#include <map>
#include <vector>
#include <functional>
#include <memory>
#include <typeinfo>
#include <typeindex>
#include <type_traits>
#include <iostream>
#include "Item.h"
#include "Context.h"

namespace PyNet::DI {
    class ContextBuilder {

    private:
        // Factory signature
        template <class InstanceType, class... Args> 
        using FactoryFunction = InstanceType * (*)(Args...);

        std::shared_ptr<ItemContainer> _container = std::make_shared<ItemContainer>();
        std::shared_ptr<Context> _context = std::make_shared<Context>(_container);

    public:

        ContextBuilder() {
            AddInstance(_context.get(), InstanceMode::Shared);
        }

        // Add an already instantiated object to the context
        template <typename InstanceType>
        ContextBuilder* AddInstance(InstanceType* instance, InstanceMode instanceMode)
        {
            if (instance == nullptr)
                throw std::runtime_error(std::string("Trying to add nullptr instance for type: ") + typeid(InstanceType).name());

            auto& item = _container->GetInitialItem<InstanceType>();

            if (item.instancePtr)
                throw std::runtime_error(std::string("Instance already in Context for type: ") + typeid(InstanceType).name());

            if (instanceMode == InstanceMode::Shared) {
                item.instancePtr = std::make_shared<void*>();
                *item.instancePtr = static_cast<void*>(instance);
            }

            return this;
        }

        // Add a factory function to context builder
        template <class InstanceType, class... Args>
        void AddFactoryPriv(FactoryFunction<InstanceType, Args...> factoryFunction, InstanceMode instanceMode = InstanceMode::Shared)
        {
            auto& item = _container->GetInitialItem<InstanceType>();

            if (item.factory)
                throw std::runtime_error(std::string("Factory already registed for type: ") + typeid(InstanceType).name());

            item.factory = [factoryFunction, instanceMode](Context& context)
            {
                return factoryFunction(context.GetShared<typename Args::element_type>()...);
            };

            if (instanceMode == InstanceMode::Shared) {
                item.instancePtr = std::make_shared<void*>();
                *item.instancePtr = item.factory(*_context);
            }
 
            item.instanceMode = instanceMode;
        }

        template <typename InstanceType>
        void AddFactoryPriv(InstanceType)
        {
            // Use a dummy is_void type trait to force GCC to display instantiation type in error message
            static_assert(std::is_void<InstanceType>::value, "Factory has incorrect signature, should take (const) references and return a pointer! Examlpe: Foo* Foo::factory(Bar& bar); ");
        }

        // Variadic template to add a list of classes with factory methods
        template <typename InstanceType1, typename InstanceType2, typename... ITs>
        ContextBuilder* AddClass(InstanceMode instanceMode = InstanceMode::Shared)
        {
            AddFactoryPriv(InstanceType1::factory, instanceMode);
            AddClass<InstanceType2, ITs...>();
            return this;
        }

        template <typename InstanceTypeLast>
        ContextBuilder* AddClass(InstanceMode instanceMode = InstanceMode::Shared)
        {
            AddFactoryPriv(InstanceTypeLast::factory, instanceMode);
            return this;
        }

        std::shared_ptr<Context> Build() {
            return _context;
        }
    };
}
