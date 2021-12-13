#pragma once

#include "PyNet.DI/ContextBuilder.h"
#include "PyNet.Models.Cuda/CudaMatrix.h"
#include "PyNet.Models.Cuda/CudaVector.h"
#include <memory>

class UnitTest
{
private:
	std::shared_ptr<PyNet::DI::Context> _context;
	std::unique_ptr<PyNet::DI::ContextBuilder> _builder;
public:

	UnitTest() {

		_builder = std::make_unique<PyNet::DI::ContextBuilder>();

		_builder
			->AddClass<CudaMatrix>(PyNet::DI::InstanceMode::Unique)
			->AddClass<CudaVector>(PyNet::DI::InstanceMode::Unique);
		
		_context = _builder->Build();
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

