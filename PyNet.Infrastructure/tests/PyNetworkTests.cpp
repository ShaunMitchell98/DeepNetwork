#include "CppUnitTest.h"
#include "PyNetwork.h"
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
			auto network = GetUniqueService<PyNetwork>();
		
			network->AddLayer(2);
			network->AddLayer(2);

			double weights[4] = { 1, 2, 3, 4 };
			//network._weights[0].get().Set(2, 2, weights);

			auto output = network->Run(input);

			Assert::AreEqual(0.48808328584886568, output[0]);
			Assert::AreEqual(0.51191671415113438, output[1]);
		}

		TEST_METHOD(Train_WhenCalled_TrainsNetwork)
		{
			auto input = new double[2] {1, 1};

			auto network = GetUniqueService<PyNetwork>();
			network->AddLayer(2);
			network->AddLayer(2);

			auto output = new double[2]{ 1, 0 };

			network->Train(&input, &output, 1, 1, 0.1, 0, 1);

			delete[] input;

			Assert::AreEqual(0.48808328584886568, output[0]);
			Assert::AreEqual(0.51191671415113438, output[1]);

			delete[] output;
		}
	};
}
