module;
#include "CppUnitTest.h"
export module PyNet.Infrastructure.Tests:PyNetworkTests;

import PyNet.DI;
import PyNet.Models;
import PyNet.Infrastructure;
import :UnitTest;

using namespace PyNet::Infrastructure;
using namespace PyNet::Models;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace PyNet::Infrastructure::Tests
{
	TEST_CLASS(PyNetworkTests), public UnitTest
	{
	public:

		TEST_METHOD(Run_WhenCalled_ReturnsOutput)
		{
			double input[4] = { 1, 1, 1, 1 };

			auto context = PyNetwork_Initialise(false, true);
			PyNetwork_AddLayer(context, 2, ActivationFunctionType::Logistic);
			PyNetwork_AddLayer(context, 2, ActivationFunctionType::Logistic);

			double weights[4] = { 1, 2, 3, 4 };
			//network._weights[0].get().Set(2, 2, weights);

			auto output = PyNetwork_Run(context, input);

			Assert::AreEqual(0.48808328584886568, output[0]);
			Assert::AreEqual(0.51191671415113438, output[1]);
		}

		TEST_METHOD(Train_WhenCalled_TrainsNetwork)
		{
			auto input = new double[2] {1, 1};

			auto context = PyNetwork_Initialise(false, true);
			PyNetwork_AddLayer(context, 2, ActivationFunctionType::Logistic);
			PyNetwork_AddLayer(context, 2, ActivationFunctionType::Logistic);

			auto output = new double[2]{ 1, 0 };

			PyNetwork_Train(context, &input, &output, 1, 1, 0.1, 0, 1);

			delete[] input;

			Assert::AreEqual(0.48808328584886568, output[0]);
			Assert::AreEqual(0.51191671415113438, output[1]);

			delete[] output;
		}
	};
}
