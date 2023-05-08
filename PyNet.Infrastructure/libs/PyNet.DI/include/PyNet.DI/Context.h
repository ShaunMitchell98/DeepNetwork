#pragma once
#include <memory>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <any>
#include "ItemContainer.h"
#include <concepts>
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

template<typename T>
concept Shared = requires(T a) {
    {a} -> std::convertible_to<shared_ptr<typename T::element_type>>;
};

using namespace std;

namespace PyNet::DI {

    class Context
    {
    private:
        shared_ptr<ItemContainer> _container;
    public:
        Context(shared_ptr<ItemContainer> container) : _container{ container } {}

        void MakeReferencesWeak() {
            _container->MakeReferencesWeak();
        }

        static auto factory(shared_ptr<ItemContainer> container) {
            return new Context{ container };
        }

        template <class RequiredType>
        unique_ptr<RequiredType> GetUnique() const
        {
            auto name = string(typeid(RequiredType).name());
            return GetUnique<RequiredType>(name);
        }

        template <class RequiredType>
        unique_ptr<RequiredType> GetUnique(string& typeName) const
        {
            auto& item = _container->GetItem<RequiredType>(typeName);
            item.Marker = true;
            any cast = any(*this);
            auto temp = static_cast<RequiredType*>(item.GetInstance(cast));
            item.Marker = false;

            auto result = unique_ptr<RequiredType>(temp);
            return result;
        }

        template <class RequiredType>
        shared_ptr<RequiredType> GetShared() const
        {
            auto name = string(typeid(RequiredType).name());
            return GetShared<RequiredType>(name);
        }

        template <class RequiredType>
        shared_ptr<RequiredType> GetShared(string& typeName) const
        {
            auto& item = _container->GetItem<RequiredType>(typeName);

            shared_ptr<RequiredType> sharedPtr;

            if (!item.HasInstance() && !item.Marker)
            {
                auto cast = any(*this);
                sharedPtr = item.GenerateInstance(cast);

                if (item.Marker)
                {
                    throw runtime_error(string("Cyclic dependecy while instantiating type: ") + typeid(RequiredType).name());
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