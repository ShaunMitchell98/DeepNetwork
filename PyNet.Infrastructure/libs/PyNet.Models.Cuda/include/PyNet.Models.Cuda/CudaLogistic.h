#pragma once
#include "PyNet.Models/Logistic.h"
#include "PyNet.DI/Context.h"

using namespace std;

namespace PyNet::Models::Cuda {

	class __declspec(dllexport) CudaLogistic : public Logistic {

		shared_ptr<PyNet::DI::Context> _context;

	public:

		static auto factory(shared_ptr<PyNet::DI::Context> context) {
			return new CudaLogistic{ context };
		}

		CudaLogistic(shared_ptr<PyNet::DI::Context> context) : _context{ context } {}

		typedef Activation base;

		void Apply(Matrix& input) override;

		unique_ptr<Matrix> CalculateDerivative(const Matrix& input) override;

		~CudaLogistic() override = default;
	};
}
