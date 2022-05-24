#pragma once
#include <typeindex>
#include <functional>
#include <any>
#include "ItemContainer.h"
#include "Context.h"

using namespace std;

namespace PyNet::DI {

	template<typename InputType>
	class ItemRegistrar {

	private:
		type_index _type;
		ItemContainer& _container;

		template <class... Args>
		using FactoryFunction = InputType * (*)(Args...);

		template <class OutputType, class... Args, typename enable_if<is_base_of<OutputType, InputType>::value>::type* = nullptr>
		void RegisterFactory(Item<OutputType>& item, FactoryFunction<Args...> factoryFunction) noexcept
		{
			item.SetFactory([factoryFunction](any& input)
			{
				auto context = any_cast<Context>(input);
				return static_cast<OutputType*>(factoryFunction(context.Get<Args>()...));
			});
		}

	public:

		ItemRegistrar(type_index type, ItemContainer& container) : _type{ type }, _container{ container } {}

		ItemRegistrar<InputType>& AsSelf() {
			auto& item = _container.Add<InputType>();
			RegisterFactory(item, InputType::factory);
	
			return *this;
		}

		template<typename OutputType>
		ItemRegistrar<InputType>& As() {
			auto& item = _container.Add<OutputType>();
			RegisterFactory(item, InputType::factory);
			return *this;
		}
	};
}