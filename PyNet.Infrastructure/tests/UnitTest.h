#pragma once

#include <Setup.h>

namespace PyNet::Infrastructure::Tests {

	class UnitTest
	{
	private:
		std::shared_ptr<PyNet::DI::Context> _context;
	public:

		UnitTest() {
			_context = std::move(GetContext(true, false));
		}

		template<class T>
		std::unique_ptr<T> GetUniqueService() {
			return _context->GetUnique<T>();
		}

		template<class T>
		std::shared_ptr<T> GetSharedService() {
			return _context->GetShared<T>();
		}
	};
}




