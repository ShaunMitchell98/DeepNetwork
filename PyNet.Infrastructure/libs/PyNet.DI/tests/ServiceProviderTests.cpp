#include <gtest/gtest.h>
#include <memory>
#include "PyNet.DI/ServiceProvider.h"
#include "PyNet.DI/ServiceDescriptor.h"
#include <cmath>

using namespace std;
using namespace PyNet::DI;

namespace PyNet::DI::Tests
{
    class TestClass
    {
        public:
        int Value = 5;
    };

    TEST(ServiceProviderTests, Provider_GivenContainsService_ReturnsService)
    {
        auto descriptors = vector<IServiceDescriptor*>();

        auto factory = [](IServiceProvider& input) -> shared_ptr<void>
        {
            return shared_ptr<void>(new TestClass(), [](void* ptr) { delete static_cast<TestClass*>(ptr); });
        };

        auto descriptor = make_unique<ServiceDescriptor<TestClass>>(factory);
        descriptors.push_back(descriptor.get());

        auto provider = make_unique<ServiceProvider>(descriptors);

        auto instance = provider->GetService<TestClass>();

        ASSERT_EQ(5, instance->Value);
    }
}