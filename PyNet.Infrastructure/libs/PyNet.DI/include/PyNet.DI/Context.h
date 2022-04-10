#pragma once
#include <memory>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <any>
#include "ItemContainer.h"
#include "Item.h"

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
        shared_ptr<ItemContainer> _container;
    public:
        Context(shared_ptr<ItemContainer> container) : _container{ container } {}

        static auto factory(shared_ptr<ItemContainer> container) {
            return new Context{ container };
        }

        template <class T>
        unique_ptr<T> GetUnique()
        {
            auto& item = _container->GetItem<T>();

            item.marker = true;
            any cast = any(*this);
            T* temp = static_cast<T*>(item.factory(cast));
            item.marker = false;

            auto result = unique_ptr<T>(temp);
            return move(result);
        }

        template <class T>
        shared_ptr<T> GetShared()
        {
            Item& item = _container->GetItem<T>();

            if (!item.instancePtr)
            {
                auto cast = any(*this);
                item.instancePtr = make_shared<void*>();
                *item.instancePtr = item.factory(cast);
            }

            if (item.marker) {
                throw runtime_error(string("Cyclic dependecy while instantiating type: ") + typeid(T).name());
            }

            T* elementPtr = static_cast<T*>(*item.instancePtr);
            return shared_ptr<T>(item.instancePtr, elementPtr);
        }
    };
}