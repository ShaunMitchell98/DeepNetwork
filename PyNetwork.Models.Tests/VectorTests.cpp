#include "CppUnitTest.h"
#include <vector>
#include "PyNet.Models/Vector.h"
#include <memory>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace PyNet::Models::Tests
{
	TEST_CLASS(VectorTests)
	{
	public:

		TEST_METHOD(Constructor_GivenDoubleArray_ReturnsNetworkVectorWithCorrectValues)
		{
			double testArray[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };

			auto myVector = std::make_unique<PyNet::Models::Vector>(10, testArray, ActivationFunctionType::Logistic);

			for (auto i = 0; i < 10; i++) {
				Assert::AreEqual(testArray[i], myVector->GetValue(i));
			}

			Assert::AreEqual(10, myVector->Rows);
			Assert::AreEqual(1, myVector->Cols);
		}
	};
}
