#include "UnitTest.h"
#include "PyNet.Models.Cpu/CpuMatrix.h"
#include "../Src/Layers/InputLayer.h"

using namespace PyNet::Models::Cpu;
using namespace PyNet::DI;
using namespace PyNet::Infrastructure::Layers;

namespace PyNet::Infrastructure::Tests::Layers {

	class InputLayerTests : public UnitTest {};

	TEST_F(InputLayerTests, Layer_ApplyCalled_ReturnsInput) {

		auto matrix = make_shared<CpuMatrix>();
		matrix->Initialise(500, 300, true);

		auto inputLayer = GetSharedService<InputLayer>();
		inputLayer->Initialise(matrix->GetRows(), matrix->GetCols());
		inputLayer->SetInput(matrix->GetValues().data());
		auto output = inputLayer->Apply(matrix);

		ASSERT_EQ(matrix->GetRows(), output->GetRows());
		ASSERT_EQ(matrix->GetCols(), output->GetCols());

		for (size_t row = 1; row <= matrix->GetRows(); row++) {
			for (size_t col = 1; col <= matrix->GetCols(); col++) {
				ASSERT_EQ((*matrix)(row, col), (*output)(row, col));
			}
		}
	}

	TEST_F(InputLayerTests, Layer_dLossDInputCalled_DoesNotThrow) {

		auto dLoss_dOutput = make_shared<CpuMatrix>();
		dLoss_dOutput->Initialise(500, 300, true);

		auto inputLayer = GetSharedService<InputLayer>();
		inputLayer->Initialise(dLoss_dOutput->GetRows(), dLoss_dOutput->GetCols());
		inputLayer->SetInput(dLoss_dOutput->GetValues().data());

		ASSERT_NO_THROW(inputLayer->dLoss_dInput(*dLoss_dOutput));
	}
}