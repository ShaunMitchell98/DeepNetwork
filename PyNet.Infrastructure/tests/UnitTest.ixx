module;
#include <memory>
export module PyNet.Infrastructure.Tests:UnitTest;

import PyNet.Infrastructure;
import PyNet.DI;

using namespace std;
using namespace PyNet::DI;
using namespace PyNet::Infrastructure;

class UnitTest
{
private:
	shared_ptr<Context> _context;
public:

	UnitTest() {
		_context = move(GetContext(true, false));
	}

	template<class T>
	unique_ptr<T> GetUniqueService() {
		return _context->GetUnique<T>();
	}

	template<class T>
	shared_ptr<T> GetSharedService() {
		return _context->GetShared<T>();
	}
};