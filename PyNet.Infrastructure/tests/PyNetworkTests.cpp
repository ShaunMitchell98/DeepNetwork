#include "CppUnitTest.h"
#include <vector>
#include "PyNetwork.h"
#include "Logger.h"
#include <memory>
#include "UnitTest.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace PyNet::Infrastructure::Tests
{
	TEST_CLASS(PyNetworkTests), public UnitTest
	{
	public:

		TEST_METHOD(Run_WhenCalled_ReturnsOutput)
		{
			double input[4] = { 1, 1, 1, 1 };
			auto context = GetContext(false);
			auto network = context->get<PyNetwork>();
		
			network->AddLayer(2, ActivationFunctions::ActivationFunctionType::Logistic);

			network->Weights = std::vector<std::shared_ptr<PyNet::Models::Matrix>>();

			double weights[4] = { 1, 2, 3, 4 };
			network->Weights.push_back(std::make_shared<PyNet::Models::Matrix>(2, 2, weights));

			auto output = network->Run(input);

			Assert::AreEqual(0.48808328584886568, output[0]);
			Assert::AreEqual(0.51191671415113438, output[1]);
		}

		TEST_METHOD(Train_WhenCalled_TrainsNetwork)
		{
			auto input = new double[2];
			for (int i = 0; i < 2; i++) {
				input[i] = 1;
			}

			auto context = GetContext(false);
			auto network = context->get<PyNetwork>();
			network->AddLayer(2, ActivationFunctions::ActivationFunctionType::Logistic);

			auto output = new double[2];

			output[0] = 1;
			output[1] = 0;

			network->Train(&input, &output, 1, 1, 0.1);

			delete[] input;

			Assert::AreEqual(0.48808328584886568, output[0]);
			Assert::AreEqual(0.51191671415113438, output[1]);

			delete[] output;
		}
	};
}
