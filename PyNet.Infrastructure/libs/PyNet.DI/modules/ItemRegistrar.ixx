module;
#include <typeindex>
#include <functional>
#include <any>
export module PyNet.DI:ItemRegistrar;

import :ItemContainer;
import :Context;

using namespace std;

namespace PyNet::DI {

	template<typename T>
	class ItemRegistrar {

	private:
		type_index _type;
		ItemContainer& _container;

		template <class InstanceType, class... Args>
		using FactoryFunction = InstanceType * (*)(Args...);

		template <class InstanceType, class... Args>
		void RegisterFactory(Item& item, FactoryFunction<InstanceType, Args...> factoryFunction)
		{
			item.factory = [factoryFunction](any& input)
			{
				auto context = any_cast<Context>(input);
				return factoryFunction(context.GetShared<typename Args::element_type>()...);
			};
		}

	public:

		ItemRegistrar(type_index type, ItemContainer& container) : _type{ type }, _container{ container } {}

		ItemRegistrar<T>& AsSelf() {
			auto& item = _container.RegisterItem<T>();
			item.type = _type;
			RegisterFactory(item, T::factory);
	

			return *this;
		}

		template<typename U>
		ItemRegistrar<T>& As() {
			auto& item = _container.RegisterItem<U>();
			item.type = _type;
			RegisterFactory(item, T::factory);
			return *this;
		}


	};
}