#include "UnitTest.h"
#include "PyNet.Models.Cpu/CpuMatrix.h"
#include "../Src/Layers/DenseLayer.h"

using namespace PyNet::Models::Cpu;
using namespace PyNet::Infrastructure::Layers;

namespace PyNet::Infrastructure::Tests::Layers {

	class DenseLayerTests : public UnitTest {};

	TEST_F(DenseLayerTests, Layer_ApplyCalled_ReturnsOutput) {

		auto input = GetSharedService<Matrix>();

		auto inputRows = 500;
		input->Initialise(inputRows, 1, true);

		auto inputValue = 0.429;

		for (auto& value : *input) {
			value = inputValue;
		}

		auto outputSize = 250;

		auto weights = GetUniqueService<Matrix>();
		auto dLoss_dWeightSum = GetUniqueService<Matrix>();
		auto tempInput = GetUniqueService<Matrix>();

		weights->Initialise(outputSize, inputRows, false);

		auto weightValue = 0.52;
		for (auto& weight : *weights) {
			weight = weightValue;
		}

		auto denseLayer = make_unique<DenseLayer>(GetSharedService<AdjustmentCalculator>(), move(weights), move(dLoss_dWeightSum), move(tempInput), GetUniqueService<Matrix>(),
			GetSharedService<ILogger>());

		auto output = denseLayer->Apply(input);

		ASSERT_EQ(outputSize, output->GetRows());
		ASSERT_EQ(1, output->GetCols());

		for (auto& outputValue : *output) {
			ASSERT_FLOAT_EQ(inputRows * inputValue * weightValue + denseLayer->Bias, outputValue);
		}
	}

	TEST_F(DenseLayerTests, Layer_dLossDInputCalled_ReturnsOutput) {

		auto dLoss_dOutput = GetSharedService<Matrix>();

		auto dLossdOutputRows = 500;
		dLoss_dOutput->Initialise(dLossdOutputRows, 1, true);

		auto dLossdOutputValue = 0.429;

		for (auto& value : *dLoss_dOutput) {
			value = dLossdOutputValue;
		}

		auto outputSize = 250;

		auto weights = GetUniqueService<Matrix>();
		auto dLoss_dWeightSum = GetUniqueService<Matrix>();
		auto tempInput = GetUniqueService<Matrix>();

		weights->Initialise(outputSize, dLossdOutputRows, false);

		auto weightValue = 0.52;
		for (auto& weight : *weights) {
			weight = weightValue;
		}

		auto denseLayer = make_unique<DenseLayer>(GetSharedService<AdjustmentCalculator>(), move(weights), move(dLoss_dWeightSum), move(tempInput), GetUniqueService<Matrix>(),
			GetSharedService<ILogger>());

		auto dLoss_dInput = denseLayer->dLoss_dInput(*dLoss_dOutput);

		ASSERT_EQ(dLossdOutputRows, dLoss_dInput->GetRows());
		ASSERT_EQ(1, dLoss_dInput->GetCols());

		for (auto& dLossdInputValue : *dLoss_dInput) {
			ASSERT_FLOAT_EQ(outputSize * weightValue * dLossdOutputValue, dLossdInputValue);
		}
	}
}