#pragma once
#include "ServiceCollection.h"

using namespace std;

namespace PyNet::DI {

	template<typename ImplementationType>
	class ItemRegistrar {

	private:
		ServiceCollection& _serviceCollection;

	public:

		ItemRegistrar(ServiceCollection& serviceCollection) : _serviceCollection{ serviceCollection } {}

		ItemRegistrar<ImplementationType>& AsSelf() {
			_serviceCollection.Add<ImplementationType, ImplementationType>();
			return *this;
		}

		template<typename ServiceType>
		ItemRegistrar<ImplementationType>& As() {
			_serviceCollection.Add<ServiceType, ImplementationType();
			return *this;
		}
	};
}