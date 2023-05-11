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
#include "ServiceCollection.h"
#include "InstanceMode.h"
#include "ItemRegistrar.h"

using namespace std;

namespace PyNet::DI {
    class ContextBuilder {

    private:
        unique_ptr<ServiceCollection> _serviceCollection;
        shared_ptr<Context> _context;

    public:

        ContextBuilder() : _serviceCollection{ make_unique<ServiceCollection>()} {
            _context = make_shared<Context>(_serviceCollection);
            RegisterInstance(_context, InstanceMode::Shared);
        }

        template <class InstanceType>
        void RegisterInstance(shared_ptr<InstanceType> instance, InstanceMode instanceMode) const
        {
            if (instance == nullptr)
                throw runtime_error(string("Trying to add nullptr instance for type: ") + typeid(InstanceType).name());

            auto& item = _serviceCollection->Add<InstanceType>(instance);
        }

        template <typename InstanceType>
        unique_ptr<ItemRegistrar<InstanceType>> RegisterType(InstanceMode instanceMode = InstanceMode::Shared) const
        {
            return make_unique<ItemRegistrar<InstanceType>>(*_serviceCollection);
        }

        shared_ptr<ServiceProvider> Build() {

            return make_shared<ServiceProvider>(_serviceCollection->Descriptors);
        }
    };
}