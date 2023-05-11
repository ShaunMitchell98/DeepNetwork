#pragma once
#include <map>
#include <vector>
#include <typeindex>
#include <stdexcept>
#include <type_traits>
#include "IServiceDescriptor.h"
#include "ServiceDescriptor.h"

using namespace std;

namespace PyNet::DI
{

    class IServiceCollection
    {
        public:
        vector<IServiceDescriptor*> Descriptors;

        template<class ServiceType, class ImplementationType>
        virtual ServiceDescriptor<ServiceType>& Add(shared_ptr<ServiceType> implementationInstance) = 0;

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