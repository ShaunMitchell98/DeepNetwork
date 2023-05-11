#pragma once

#include <memory>

using namespace std;

namespace PyNet::DI
{
    class IServiceProvider
    {
        public:
        virtual ~IServiceProvider() = default;

        template <typename T>
        shared_ptr<T> GetService()
        {
            auto service = GetServiceInternal(typeid(T));
            if (service == nullptr)
            {
                throw runtime_error("Service not found.");
            }
            return static_pointer_cast<T>(service);
        }

        private:
        virtual shared_ptr<void> GetServiceInternal(const type_info& serviceType) = 0;
    };
}