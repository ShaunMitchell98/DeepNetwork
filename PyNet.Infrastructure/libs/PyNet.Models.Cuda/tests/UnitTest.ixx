module;
#include <memory>
export module PyNet.Models.Cuda.Tests:UnitTest;

import PyNet.DI;
import PyNet.Models.Cuda;

using namespace PyNet::DI;
using namespace std;
using namespace PyNet::Models::Cuda;

class UnitTest
{
private:
	shared_ptr<Context> _context;
public:

	UnitTest() {

		auto builder = make_unique<ContextBuilder>();

		auto cudaModule = make_unique<CudaModule>();
		cudaModule->Load(*builder);
		
		_context = builder->Build();
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