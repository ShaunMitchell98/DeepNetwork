#include "ServiceDescriptor.h"
#include "ServiceProvider.h"
#include <functional>
#include <any>

using namespace PyNet::DI;

namespace PyNet::DI::Internal
{
	template<typename ImplementationType>
	class ServiceFactory 
	{
		template <class... Args>
		using FactoryFunction = ImplementationType * (*)(Args...);

		template <class ServiceType, class... Args, typename enable_if<is_base_of<ServiceType, ImplementationType>::value>::type* = nullptr>
		function<any* (any&)> RegisterFactory(FactoryFunction<Args...> factoryFunction) noexcept
		{
			return [factoryFunction](any& input)
			{
				auto serviceProvider = any_cast<ServiceProvider>(input);
				return static_cast<ServiceType*>(factoryFunction(serviceProvider.GetService<Args>()...));
			};
		}
	};
}