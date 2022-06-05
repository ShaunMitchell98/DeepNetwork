#include "UnitTest.h"
#include "PyNet.Models.Cpu/CpuMatrix.h"
#include "../Src/Layers/InputLayer.h"
#include "Layers/FakeLayer.h"
#include "../Src/NetworkTrainer.h"

using namespace PyNet::Models::Cpu;
using namespace PyNet::DI;
using namespace PyNet::Infrastructure::Layers;

namespace PyNet::Infrastructure::Tests
{
	class NetworkTrainerTests : public UnitTest {};

	TEST_F(NetworkTrainerTests, NetworkTrainer_GivenTrainingPair_DoesNotThrow)
	{
		auto pyNetwork = GetSharedService<PyNetwork>();

		double inputValues[5] = { 1.2, 2.4, 3.9, 4.3, 5.2 };

		auto input = GetSharedService<Matrix>();
		input->Initialise(5, 1);
		*input = inputValues;

		auto inputLayer = GetUniqueService<InputLayer>();
		inputLayer->Initialise(5, 1);

		pyNetwork->Layers.push_back(move(inputLayer));

		auto fakeLayer = GetUniqueService<FakeLayer>();

		auto fakeValue = 5.0;

		fakeLayer->SetValue(fakeValue);

		pyNetwork->Layers.push_back(move(fakeLayer));

		auto networkTrainer = GetUniqueService<NetworkTrainer>();

		auto trainingPairs = vector<pair<shared_ptr<Matrix>, shared_ptr<Matrix>>>();

		auto inputMatrix = shared_ptr<Matrix>(input);

		auto expectedOutputMatrix = shared_ptr<Matrix>(input);

		auto settings = GetSharedService<Settings>();
		settings->RunMode = RunMode::Training;
		settings->Momentum = 0.2;
		settings->NewBatch = true;
		settings->BatchSize = 3;
		settings->BaseLearningRate = 0.1;
		settings->Epochs = 2;

		pair<shared_ptr<Matrix>, shared_ptr<Matrix>> pair;
		pair.first = inputMatrix;
		pair.second = expectedOutputMatrix;

		trainingPairs.push_back(pair);
		
		ASSERT_NO_THROW(networkTrainer->TrainNetwork(trainingPairs));
	}
}