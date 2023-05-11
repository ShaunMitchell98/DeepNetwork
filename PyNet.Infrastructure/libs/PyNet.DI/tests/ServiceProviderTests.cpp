#include <gtest/gtest.h>
#include <memory>
#include "PyNet.DI/ServiceProvider.h"
#include "PyNet.DI/ServiceDescriptor.h"
#include <cmath>

using namespace std;
using namespace PyNet::DI;

namespace PyNet::DI::Tests
{
    class BaseClass
    {
        public:
        int Value = 5;
    };

    class DerivedClass1 : public BaseClass
	{
        public:
        DerivedClass1() 
		{
            Value = 1;
        }
    };

    class DerivedClass2 : public BaseClass 
    {
        public:
        DerivedClass2() 
        {   
            Value = 2;
        }
    };

    TEST(ServiceProviderTests, Provider_GivenContainsService_ReturnsService)
    {
        auto descriptors = vector<IServiceDescriptor*>();

        auto factory = [](IServiceProvider& input) -> shared_ptr<void>
        {
            return shared_ptr<void>(new BaseClass(), [](void* ptr) { delete static_cast<BaseClass*>(ptr); });
        };

        auto descriptor = make_unique<ServiceDescriptor<BaseClass>>(factory);
        descriptors.push_back(descriptor.get());

        auto provider = make_unique<ServiceProvider>(descriptors);

        auto instance = provider->GetService<BaseClass>();

        ASSERT_EQ(5, instance->Value);
    }

    TEST(ServiceProviderTests, Provider_GivenMultipleImplementations_ReturnsMostRecentlyAddedImplementation)
    {
        auto descriptors = vector<IServiceDescriptor*>();

        auto factory1 = [](IServiceProvider& input) -> shared_ptr<void>
        {
            return shared_ptr<void>(new DerivedClass1(), [](void* ptr) { delete static_cast<DerivedClass1*>(ptr); });
        };

        auto factory2 = [](IServiceProvider& input) -> shared_ptr<void>
        {
            return shared_ptr<void>(new DerivedClass2(), [](void* ptr) { delete static_cast<DerivedClass2*>(ptr); });
        };

        auto descriptor1 = make_unique<ServiceDescriptor<BaseClass>>(factory1);
        descriptors.push_back(descriptor1.get());

        auto descriptor2 = make_unique<ServiceDescriptor<BaseClass>>(factory2);
        descriptors.push_back(descriptor2.get());

        auto provider = make_unique<ServiceProvider>(descriptors);

        auto instance = provider->GetService<BaseClass>();

        ASSERT_EQ(2, instance->Value);
    }

    TEST(ServiceProviderTests, Provider_GivenDoesNotContainService_Throws)
    {
        auto descriptors = vector<IServiceDescriptor*>();

        auto provider = make_unique<ServiceProvider>(descriptors);

        ASSERT_THROW({
                try
                {
                    auto instance = provider->GetService<BaseClass>();
                }
                catch (runtime_error& ex)
                {
                    ASSERT_STREQ("Service not found.", ex.what());
                    throw;
                }
            }, runtime_error);
    }
}