#include <vector>
#include "IServiceDescriptor.h"
#include "IServiceProvider.h"

using namespace std;

namespace PyNet::DI
{
	class ServiceProvider : public IServiceProvider
	{
		public:

		ServiceProvider(vector<IServiceDescriptor*> descriptors) 
		{
			_descriptors = descriptors;
		}

		private:

		vector<IServiceDescriptor*> _descriptors;

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