#pragma once
#include <memory>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <any>
#include "ServiceProvider.h"
/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Gyorgy Szekely
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

using namespace std;

namespace PyNet::DI {

    class Context
    {
    private:
        shared_ptr<ServiceProvider> _serviceProvider;
    public:
        Context(shared_ptr<ServiceProvider> serviceProvider) : _serviceProvider{ serviceProvider } {}

        static auto factory(shared_ptr<ServiceProvider> serviceProvider) {
            return new Context{ serviceProvider };
        }

        template <class ServiceType>
        unique_ptr<ServiceType> GetUnique() const
        {
            auto name = string(typeid(ServiceType).name());
            return GetUnique<ServiceType>(name);
        }

        template <class ServiceType>
        unique_ptr<ServiceType> GetUnique(string& typeName) const
        {
            auto& item = _serviceProvider->GetService<ServiceType>(typeName);
            item.Marker = true;
            any cast = any(*this);
            auto temp = static_cast<ServiceType*>(item.GetInstance(cast));
            item.Marker = false;

            auto result = unique_ptr<ServiceType>(temp);
            return result;
        }

        template <class ServiceType>
        shared_ptr<ServiceType> GetShared() const
        {
            auto name = string(typeid(ServiceType).name());
            return GetShared<ServiceType>(name);
        }

        template <class ServiceType>
        shared_ptr<ServiceType> GetShared(string& typeName) const
        {
            auto& item = _serviceProvider->GetItem<ServiceType>(typeName);

            shared_ptr<ServiceType> sharedPtr;

            if (!item.HasInstance() && !item.Marker)
            {
                auto cast = any(*this);
                sharedPtr = item.GenerateInstance(cast);

                if (item.Marker)
                {
                    throw runtime_error(string("Cyclic dependency while instantiating type: ") + typeid(ServiceType).name());
                }

                return sharedPtr;
            }

            return item.GetShared();
        }

        template <Shared T>
        T Get() {
            return GetShared<typename T::element_type>();
        }

        template <class T>
        T Get(...) {
            return GetUnique<typename T::element_type>();
        }
    };
}