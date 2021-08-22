#include "CppUnitTest.h"
#include <vector>
#include "PyNet.Models/Logistic.h"
#include <memory>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace DeepNetworkTests
{
	TEST_CLASS(LogisticTests)
	{
	public:

		TEST_METHOD(ApplyFunction_GivenVectorOfZeros_ReturnsCorrectValues)
		{
			double testArray[5] = { 0, 0, 0, 0, 0};

			auto values = std::vector<double>();

			for (int i = 0; i < 5; i++) {
				values.push_back(testArray[i]);
			}

			auto logisticFunction = std::make_unique<ActivationFunctions::Logistic>();
			logisticFunction->Apply(values);
			

			for (int j = 0; j < 5; j++) {
				Assert::AreEqual(0.5, values[j]);
			}
		}
	};
}
