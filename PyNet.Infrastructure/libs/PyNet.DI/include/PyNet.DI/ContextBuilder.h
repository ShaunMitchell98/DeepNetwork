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

        // Add factory method automatically if present in class
        template <typename T, typename std::enable_if< std::is_function<decltype(T::factory)>::value >::type* = nullptr>
        void AddClassAuto(void*) // argument only used to disambiguate from vararg version
        {
            AddFactoryPriv(T::factory);
        }

        template<typename T>
        void AddClassAuto(...)
        {
            throw std::runtime_error(std::string("Class {} has no factory in context!") + typeid(T).name());
        }

    public:

        // Variadic template to add a list of free standing factory functions
        template <typename T1, typename T2, typename... Ts>
        void AddFactory(T1 t1, T2 t2, Ts... ts)
        {
            AddFactoryPriv(t1, _context);
            AddFactory(t2, ts...);
        }

        template <typename T>
        void AddFactory(T t)
        {
            AddFactoryPriv(t, _context);
        }

        // Add an already instantiated object to the context
        template <typename InstanceType>
        void AddInstance(InstanceType* instance, InstanceMode instanceMode)
        {
            if (instance == nullptr)
                throw std::runtime_error(std::string("Trying to add nullptr instance for type: ") + typeid(InstanceType).name());

            Item& item = _container->GetInitialItem<InstanceType>();

            if (item.instancePtr.index() != 0)
                throw std::runtime_error(std::string("Instance already in Context for type: ") + typeid(InstanceType).name());

            std::any value = *instance;
            if (instanceMode == InstanceMode::Unique) {
                item.instancePtr = std::unique_ptr<std::any>(&value);
            }
            else {
                item.instancePtr = std::shared_ptr<std::any>(&value);
            }
        }

        // Add a factory function to context builder
        template <class InstanceType, class... Args>
        void AddFactoryPriv(FactoryFunction<InstanceType, Args...> factoryFunction, InstanceMode instanceMode = InstanceMode::Shared)
        {
            auto& item = _container->GetInitialItem<InstanceType>();

            if (item.factory)
                throw std::runtime_error(std::string("Factory already registed for type: ") + typeid(InstanceType).name());

            item.factory = [factoryFunction, instanceMode, this]()
            {
                AddInstance(factoryFunction(_context->Get<Args>()...), instanceMode);
            };

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
            AddInstance(_context.get(), InstanceMode::Shared);
            return _context;
        }
    };
}
