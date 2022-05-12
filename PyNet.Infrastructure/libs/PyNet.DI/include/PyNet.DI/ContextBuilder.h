#pragma once
#include <map>
#include <vector>
#include <functional>
#include <string>
#include <typeinfo>
#include <typeindex>
#include <type_traits>
#include <iostream>
#include "Context.h"
#include "ItemContainer.h"
#include "Item.h"
#include "InstanceMode.h"
#include "ItemRegistrar.h"

using namespace std;

namespace PyNet::DI {
    class ContextBuilder {

    private:
        shared_ptr<ItemContainer> _container;
        shared_ptr<Context> _context;

    public:

        ContextBuilder() : _container{ make_shared<ItemContainer>()} {
            _context = make_shared<Context>(_container);
            RegisterInstance(_context, InstanceMode::Shared);
        }

        // Add an already instantiated object to the context
        template <class InstanceType>
        void RegisterInstance(shared_ptr<InstanceType> instance, InstanceMode instanceMode) const
        {
            if (instance == nullptr)
                throw runtime_error(string("Trying to add nullptr instance for type: ") + typeid(InstanceType).name());

            auto& item = _container->Add<InstanceType>();

            if (item.HasInstance())
                throw runtime_error(std::string("Instance already in Context for type: ") + typeid(InstanceType).name());

            if (instanceMode == InstanceMode::Shared) {
                item.SetInstance(instance);
            }
        }

        template <typename InstanceType>
        unique_ptr<ItemRegistrar<InstanceType>> RegisterType(InstanceMode instanceMode = InstanceMode::Shared) const
        {
            return make_unique<ItemRegistrar<InstanceType>>(type_index(typeid(InstanceType)), *_container);
        }

        shared_ptr<Context> Build() {
            auto& item = _container->GetItem<Context>();
            return item.GetShared();
        }
    };
}