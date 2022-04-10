#pragma once

#include <map>
#include <vector>
#include <typeindex>
#include <stdexcept>
#include "Item.h"

namespace PyNet::DI {

    class ItemContainer 
    {
    private:

        // The object storage
        std::map<std::type_index, Item> _items;

        template<typename T, typename T::base* = nullptr>
        void DeclareBaseTypes(std::type_index& derivedType)
        {
            _items[std::type_index(typeid(typename T::base))].derivedType = derivedType;
            DeclareBaseTypes<typename T::base>(derivedType);
        }

        template <typename T>
        void DeclareBaseTypes(...) { }

    public:

        template<class InstanceType>
        Item& GetInitialItem() {

            auto instanceTypeIdx = std::type_index(typeid(InstanceType));

            DeclareBaseTypes<InstanceType>(instanceTypeIdx);
            //DeclareBaseTypes<InstanceType>(std::type_index(typeid(InstanceType)));
            return _items[std::type_index(typeid(InstanceType))];
        }

        template <class T>
        Item& GetItem()
        {
            auto it = _items.find(std::type_index(typeid(T)));

            if (it == _items.end())
            {
                throw std::runtime_error(std::string("No type ") + typeid(T).name() + " has been registered with the Context.");
            }
            else
            {
                auto& item = it->second;

                // fallback to derived type (no instance or factory, but a derived type is registered)
                if (!item.instancePtr && !item.factory && (item.derivedType != std::type_index(typeid(void))))
                    it = _items.find(item.derivedType);
            }

            return it->second;
        }
    };
}
