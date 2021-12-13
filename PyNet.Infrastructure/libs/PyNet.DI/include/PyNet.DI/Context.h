#pragma once

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

#include "ItemContainer.h"

namespace PyNet::DI {

    class Context
    {
    private:
        // The object storage
        std::shared_ptr<ItemContainer> _container;

    public:
        Context(std::shared_ptr<ItemContainer> container) : _container{ container} {}

        // Get an instance from the context, runs factories recursively to satisfy all dependencies
        template <class T>
        std::unique_ptr<T> GetUnique()
        {
            return Get<std::unique_ptr<T>>();
        }


        // Get an instance from the context, runs factories recursively to satisfy all dependencies
        template <class T>
        std::shared_ptr<T> GetShared()
        {
            return Get<std::shared_ptr<T>>();
        }

        template<class T>
        auto Get() 
        {

            Item& item = _container->GetItem<T>(); // may return derived type

            if (item.instancePtr.index() == 0 || item.instanceMode == InstanceMode::Unique)
            {
                if (item.marker) {
                    throw std::runtime_error(std::string("Cyclic dependecy while instantiating type: ") + typeid(T).name());
                }

                item.marker = true;

                item.instancePtr = std::monostate();
                item.factory();
                item.marker = false;
            }

            if (std::is_same<typename std::remove_cv<T>::type, std::shared_ptr<typename T::element_type>>::value) {
                auto ptr = std::get<std::shared_ptr<std::any>>(item.instancePtr);
                auto result = T();
                result.reset(std::any_cast<typename T::element_type>(ptr.get()));
                return result;
                //return std::invoke_result_t<T, typename T::element_type>(&T::reset, result, std::any_cast<typename T::element_type>(ptr.get()));
            }
            else {
                auto ptr = std::move(std::get<std::unique_ptr<std::any>>(item.instancePtr));
                auto result = T();
                result.reset(std::any_cast<typename T::element_type>(ptr.get()));
                return std::move(result);
                //return std::invoke_result_t<T, typename T::element_type>(&T::reset, result, std::any_cast<typename T::element_type>(ptr.get()));
            }
        }
    };
}
