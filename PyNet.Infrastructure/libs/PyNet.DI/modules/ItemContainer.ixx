module;
#include <map>
#include <vector>
#include <typeindex>
#include <stdexcept>
export module PyNet.DI:ItemContainer;

using namespace std;

import :Item;

namespace PyNet::DI {

    class ItemContainer 
    {
    private:

        map<type_index, Item> _items;

        template<typename T, typename T::base* = nullptr>
        void DeclareBaseTypes(type_index& derivedType)
        {
            _items[type_index(typeid(typename T::base))].derivedType = derivedType;
            DeclareBaseTypes<typename T::base>(derivedType);
        }

        template <typename T>
        void DeclareBaseTypes(...) { }

    public:

        ItemContainer() : _items{ std::map<type_index, Item>() } {}

        ~ItemContainer() {}

        static auto factory() {
            return new ItemContainer();
        }

        template<class InstanceType>
        Item& GetInitialItem() {

            auto instanceTypeIdx = type_index(typeid(InstanceType));

            DeclareBaseTypes<InstanceType>(instanceTypeIdx);
            return _items[type_index(typeid(InstanceType))];
        }

        template <class T>
        Item& GetItem()
        {
            auto it = _items.find(type_index(typeid(T)));

            if (it == _items.end())
            {
                throw runtime_error(string("No type ") + typeid(T).name() + " has been registered with the Container.");
            }
            else
            {
                auto& item = it->second;

                if (!item.instancePtr && (item.derivedType != type_index(typeid(void))))
                    it = _items.find(item.derivedType);
            }

            return it->second;
        }
    };
}
