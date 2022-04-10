#pragma once
#include <map>
#include <vector>
#include <functional>
#include <string>
#include <typeinfo>
#include <typeindex>
#include <type_traits>
#include <iostream>
#include "ItemRegistrar.h"
#include "InstanceMode.h"

using namespace std;

namespace PyNet::DI {
    class ContextBuilder {

    private:
        shared_ptr<ItemContainer> _container;

    public:

        ContextBuilder() : _container{ make_shared<ItemContainer>()} {
            auto context = new Context(_container);
            AddInstance(context, InstanceMode::Shared);
        }

        // Add an already instantiated object to the context
        template <class InstanceType>
        ContextBuilder& AddInstance(InstanceType* instance, InstanceMode instanceMode)
        {
            if (instance == nullptr)
                throw runtime_error(string("Trying to add nullptr instance for type: ") + typeid(InstanceType).name());

            auto& item = _container->RegisterItem<InstanceType>();

            if (item.instancePtr)
                throw runtime_error(std::string("Instance already in Context for type: ") + typeid(InstanceType).name());

            if (instanceMode == InstanceMode::Shared) {
                item.instancePtr = make_shared<void*>();
                *item.instancePtr = static_cast<void*>(instance);
            }

            return *this;
        }

        template <typename InstanceType>
        ItemRegistrar<InstanceType>& RegisterType(InstanceMode instanceMode = InstanceMode::Shared)
        {
            auto registrar = new ItemRegistrar<InstanceType>(type_index(typeid(InstanceType)), *_container);
            return *registrar;
        }

        shared_ptr<Context> Build() {
            auto& item = _container->GetItem<Context>();
            Context* elementPtr = static_cast<Context*>(*item.instancePtr);
            return shared_ptr<Context>(item.instancePtr, elementPtr);
        }
    };
}