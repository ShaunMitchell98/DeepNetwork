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

        template <class T>
        std::unique_ptr<T> GetUnique()
        {
            auto& item = _container->GetItem<T>();

            item.marker = true;
            void* temp = item.factory(*this);
            item.marker = false;

            auto result = std::unique_ptr<T>(static_cast<T*>(temp));
            return std::move(result);
        }

        template <class T>
        std::shared_ptr<T> GetShared()
        {
            Item& item = _container->GetItem<T>();

            if (!item.instancePtr)
            {
                throw std::runtime_error(std::string("No instance of type ") + typeid(T).name() + " has been registed with the Context.");
            }

            if (item.marker) {
                throw std::runtime_error(std::string("Cyclic dependecy while instantiating type: ") + typeid(T).name());
            }

            T* elementPtr = static_cast<T*>(*item.instancePtr);
            return std::shared_ptr<T>(item.instancePtr, elementPtr);
        }
    };
}
