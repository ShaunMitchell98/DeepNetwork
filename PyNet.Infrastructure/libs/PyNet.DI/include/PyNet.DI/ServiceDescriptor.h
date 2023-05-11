#pragma once
#include <functional>
#include <typeindex>
#include <memory>
#include <any>
#include "IServiceDescriptor.h"
#include "IServiceProvider.h"

using namespace std;

namespace PyNet::DI 
{
    template<class TServiceType>
    class ServiceDescriptor : public IServiceDescriptor
    {
    private:
        weak_ptr<TServiceType> _weakPtr;
        bool _weakReference = false;

        type_index _type = type_index(typeid(void));
        string _serviceType;
        function<shared_ptr<void>(IServiceProvider&)> _implementationFactory;

    public:
        // non-copyable, non-moveable
        ServiceDescriptor<TServiceType>() = default;
        ServiceDescriptor(const ServiceDescriptor& rhs) = delete;
        ServiceDescriptor& operator=(const ServiceDescriptor& rhs) = delete;
        ServiceDescriptor(ServiceDescriptor&& rhs) = delete;
        ServiceDescriptor& operator=(ServiceDescriptor&& rhs) = delete;

        ServiceDescriptor(shared_ptr<TServiceType> implementationInstance) : ImplementationInstance(implementationInstance) {}

        ServiceDescriptor(function<shared_ptr<void>(IServiceProvider&)> implementationFactory) : _serviceType(typeid(TServiceType).name()),
            _implementationFactory(implementationFactory)
        {}

        shared_ptr<TServiceType> ImplementationInstance;

        const std::string& GetServiceType() const
        {
            return _serviceType;
        }

        const function<shared_ptr<void>(IServiceProvider&)>& GetImplementationFactory() const
        {
            return _implementationFactory;;
        }

        void Reset() override
        {
           _weakReference ? _weakPtr.reset() : ImplementationInstance.reset();
        }

        void MakeReferenceWeak() override
        {
            _weakPtr = ImplementationInstance;
            ImplementationInstance.reset();
            _weakReference = true;
        }

        bool Marker = false;

        ~ServiceDescriptor() = default;
    };
}