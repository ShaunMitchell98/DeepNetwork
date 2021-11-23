#include "CppUnitTest.h"
#include <vector>
#include "NetworkTrainer.h"
#include "PyNet.Models/Matrix.h"
#include "UnitTest.h"
#include <memory>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace PyNet::Infrastructure::Tests
{
	TEST_CLASS(NetworkTrainerTests), public UnitTest
	{
	public:

		TEST_METHOD(Trainer_TrainNetworkCalled_)
		{
			double testArray[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };

			auto context = GetContext();
			auto& matrix = context->get<Matrix>();
			matrix.Set(10, 1, testArray);

			for (auto i = 0; i < 10; i++) {
				Assert::AreEqual(testArray[i], matrix.GetValue(i, 0));
			}

			Assert::AreEqual(10, matrix.GetRows());
			Assert::AreEqual(1, matrix.GetCols());
		}
	};
}
