module;
#include "CppUnitTest.h"
export module PyNet.Infrastructure.Tests:SteepestDescentTests;

import :UnitTest;
import PyNet.Models;

using namespace PyNet::Models;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace PyNet::Infrastructure::Tests
{
	TEST_CLASS(SteepestDescentTests), public UnitTest
	{
	public:

		TEST_METHOD(SteepestDescent_UpdateWeightsCalled_)
		{
			double testArray[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };

			auto matrix = GetUniqueService<Matrix>();
			matrix->Set(10, 1, testArray);

			for (auto i = 0; i < 10; i++) {
				Assert::AreEqual(testArray[i], (*matrix)(i, 0));
			}

			Assert::AreEqual(10, matrix->GetRows());
			Assert::AreEqual(1, matrix->GetCols());
		}
	};
}
