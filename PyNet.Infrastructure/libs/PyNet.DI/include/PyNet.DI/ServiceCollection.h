#pragma once
#include <map>
#include <vector>
#include <typeindex>
#include <stdexcept>
#include <type_traits>
#include "IServiceDescriptor.h"
#include "ServiceDescriptor.h"
#include "ServiceProvider.h"
#include "ServiceFactory.h"

using namespace std;
using namespace PyNet::DI::Internal;

namespace PyNet::DI 
{
    class ServiceCollection
    {
    public:

        vector<IServiceDescriptor*> Descriptors;

        static auto factory() {
            return new ServiceCollection();
        }

        template<class ServiceType, class ImplementationType>
        ServiceDescriptor<ServiceType>& Add(shared_ptr<ServiceType> implementationInstance)
        {
            auto descriptor = new ServiceDescriptor<ServiceType>(implementationInstance);

            Descriptors.push_back(descriptor);

            return *descriptor;
        }

        template<class ServiceType, class ImplementationType>
        ServiceDescriptor<ServiceType>& Add() 
        {
            auto implementationFactory = ServiceFactory::RegisterFactory(ImplementationType::factory);
            auto descriptor = new ServiceDescriptor<ServiceType>(implementationFactory);

            Descriptors.push_back(descriptor);

            return *descriptor;
        }
    };
}