#include <any>
#include <string>
#include <vector>
#include "IServiceDescriptor.h"
#include "IServiceProvider.h"
#include <concepts>

template<typename T>
concept Shared = requires(T a)
{
	{ a } -> std::convertible_to<shared_ptr<typename T::element_type>>;
};

using namespace std;

namespace PyNet::DI
{
	class ServiceProvider : public IServiceProvider
	{
		private:

		vector<IServiceDescriptor*> _descriptors;

		public:

		ServiceProvider(vector<IServiceDescriptor*> descriptors) 
		{
			_descriptors = descriptors;
		}

		//template<Shared ServiceType>
		//ServiceType GetService()
		//{
		//	for (auto descriptor : _descriptors)
		//	{
		//		if (descriptor->ServiceType == typeid(ServiceType).name())
		//		{
		//			if (descriptor->ImplementationInstance != nullptr)
		//			{
		//				return descriptor->ImplementationInstance;
		//			}
		//			else if (descriptor->ImplementationFactory != nullptr)
		//			{
		//				auto instance = descriptor->ImplementationFactory(*this);
		//				descriptor.ImplementationInstance = shared_ptr<any>(instance);
		//				return any_cast<ServiceType>(instance);
		//			}
		//		}
		//	}

		//	return nullptr;
		//}

		shared_ptr<void> GetServiceInternal(const type_info& serviceType)
		{
			for (auto it = _descriptors.rbegin(); it != _descriptors.rend(); it++)
			{
				auto descriptor = *it;

				if (descriptor->GetServiceType() == serviceType.name())
				{
					if (descriptor->GetImplementationFactory() != nullptr)
					{
						return descriptor->GetImplementationFactory()(*this);
					}
				}
			}

			return nullptr;
		}
	};
}