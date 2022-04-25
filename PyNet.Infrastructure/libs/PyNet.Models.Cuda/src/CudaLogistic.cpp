#include "CudaLogistic.h"
#include "Matrix_Operations.h"

using namespace std;

namespace PyNet::Models::Cuda {

	void CudaLogistic::Apply(Matrix& input) {

		matrix_logistic(input.GetCValues(), input.GetValues(), input.GetRows(), input.GetCols());
	}

	unique_ptr<Matrix> CudaLogistic::CalculateDerivative(const Matrix& input) {

		auto output = _context->GetUnique<Matrix>();
		output->Initialise(input.GetRows(), input.GetCols(), false);

		matrix_logistic_derivative(input.GetCValues(), output->GetValues(), input.GetRows(), input.GetCols());

		return move(output);
	}
}
