#pragma once
#include <functional>
#include <typeindex>
#include <memory>
#include <any>
#include "InstanceMode.h"
#include "IItem.h"

using namespace std;

namespace PyNet::DI {

    template<class RequiredType>
    class Item : public IItem
    {
    private:
        shared_ptr<RequiredType> _sharedPtr;
        weak_ptr<RequiredType> _weakPtr;
        bool _weakReference = false;
        function<RequiredType* (any&)> _factory;

        std::type_index _type = std::type_index(typeid(void));
        InstanceMode _instanceMode = InstanceMode::Shared;

    public:
        // non-copyable, non-moveable
        Item() = default;
        Item(const Item& rhs) = delete;
        Item& operator=(const Item& rhs) = delete;
        Item(Item&& rhs) = delete;
        Item& operator=(Item&& rhs) = delete;

        void Reset() override {
           _weakReference ? _weakPtr.reset() : _sharedPtr.reset();
        }

        shared_ptr<RequiredType> GetShared() {
            return _sharedPtr;
        }

        void* GetInstance(any& context) override {
            return _factory(context);
        }

        bool HasInstance() override {
            return _sharedPtr != nullptr;
        }

        void SetInstance(shared_ptr<RequiredType> sharedPtr) {
            _sharedPtr = sharedPtr;
        }

        void SetFactory(function<RequiredType* (any&)> factory) {
            _factory = factory;
        }

        shared_ptr<RequiredType> GenerateInstance(any& context) {
            auto sharedPtr = shared_ptr<RequiredType>(_factory(context));;
            _sharedPtr = sharedPtr;
            return sharedPtr;
        }

        void MakeReferenceWeak() override {
            _weakPtr = _sharedPtr;
            _sharedPtr.reset();
            _weakReference = true;
        }

        bool Marker = false;
    };
}