module;
#include <map>
#include <vector>
#include <functional>
#include <string>
#include <typeinfo>
#include <typeindex>
#include <type_traits>
#include <iostream>
export module PyNet.DI:ContextBuilder;

using namespace std;

import :Context;
import :ItemContainer;
import :Item;
import :InstanceMode;

export namespace PyNet::DI {
    class ContextBuilder {

    private:
        // Factory signature
        template <class InstanceType, class... Args> 
        using FactoryFunction = InstanceType * (*)(Args...);

        shared_ptr<ItemContainer> _container;

   /*     template <class Arg>
        void RegisterParameter(Item& item) {
            item.parameters.push_back(type_index(typeid(Arg)));
        }*/

        // Add a factory function to context builder
        template <class InstanceType, class... Args>
        void AddFactory(FactoryFunction<InstanceType, Args...> factoryFunction, InstanceMode instanceMode = InstanceMode::Shared)
        {
            auto& item = _container->GetInitialItem<InstanceType>();

            //if (item.factory) {
            //    string s1 = "Factory already registered for type: ";
            //    string s2 = typeid(InstanceType).name();
            //    string output = s1 + s2;
            //        throw runtime_error(output);
            //}

            //RegisterParameter<typename Args::element_type>(item)...;
          
            //item.factory = [factoryFunction, instanceMode](Context& context)
            //{
            //    return factoryFunction(context.GetShared<typename Args::element_type>()...);
            //};

            if (instanceMode == InstanceMode::Shared) {
                item.instancePtr = make_shared<void*>();
                //*item.instancePtr = item.factory(*Build());
                auto temp = Build()->GetShared<InstanceType>();
                void* temp2 = static_cast<void*>(temp.get());
                *item.instancePtr = temp2;
            }

            item.instanceMode = instanceMode;
        }

        template <typename InstanceType>
        void AddFactory(InstanceType)
        {
            // Use a dummy is_void type trait to force GCC to display instantiation type in error message
            static_assert(is_void<InstanceType>::value, "Factory has incorrect signature, should take (const) references and return a pointer! Examlpe: Foo* Foo::factory(Bar& bar); ");
        }

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

            auto& item = _container->GetInitialItem<InstanceType>();

            if (item.instancePtr)
                throw runtime_error(std::string("Instance already in Context for type: ") + typeid(InstanceType).name());

            if (instanceMode == InstanceMode::Shared) {
                item.instancePtr = make_shared<void*>();
                *item.instancePtr = static_cast<void*>(instance);
            }

            return *this;
        }

        template <typename InstanceType>
        ContextBuilder& RegisterType(InstanceMode instanceMode = InstanceMode::Shared)
        {
            AddFactory(InstanceType::factory, instanceMode);
            return *this;
        }

        shared_ptr<Context> Build() {
            auto& item = _container->GetItem<Context>();
            Context* elementPtr = static_cast<Context*>(*item.instancePtr);
            return shared_ptr<Context>(item.instancePtr, elementPtr);
        }
    };
}
