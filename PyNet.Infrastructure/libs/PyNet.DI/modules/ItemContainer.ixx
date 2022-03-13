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

    public:

        static auto factory() {
            return new ItemContainer();
        }

        template<class InstanceType>
        Item& RegisterItem() 
        {
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
            }

            return it->second;
        }
    };
}
