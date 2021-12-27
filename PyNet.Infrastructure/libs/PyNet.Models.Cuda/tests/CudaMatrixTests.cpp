#include "CppUnitTest.h"
#include "UnitTest.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace PyNet::Models::Cuda::Tests
{
	TEST_CLASS(CudaMatrixTests), public UnitTest
	{
	public:

		TEST_METHOD(Matrix_SubtractCalled_ReturnsCorrectResult)
		{
			auto first = GetUniqueService<Matrix>();
			first->Set(3, 2, new double[] { 1, 2, 3, 4, 5, 6 });

			auto second = GetUniqueService<Matrix>();
			second->Set(3, 2, new double[] { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5});

			auto result = *first - *second;

			Assert::AreEqual(3, result->GetRows());
			Assert::AreEqual(2, result->GetCols());

			Assert::AreEqual(0.5, result->GetValue(0, 0));
			Assert::AreEqual(1.5, result->GetValue(0, 1));
			Assert::AreEqual(2.5, result->GetValue(1, 0));
			Assert::AreEqual(3.5, result->GetValue(1, 1));
			Assert::AreEqual(4.5, result->GetValue(2, 0));
			Assert::AreEqual(5.5, result->GetValue(2, 1));
		}

		TEST_METHOD(Matrix_MultiplyCalled_ReturnsCorrectResult)
		{
			auto first = GetUniqueService<Matrix>();
			first->Set(3, 2, new double[] { 1, 2, 3, 4, 5, 6 });

			auto second = GetUniqueService<Matrix>();
			second->Set(2, 2, new double[] { 0.25, 0.5, 0.75, 1});

			auto result = *first * *second;

			Assert::AreEqual(3, result->GetRows());
			Assert::AreEqual(2, result->GetCols());

			Assert::AreEqual(1.75, result->GetValue(0, 0));
			Assert::AreEqual(2.5, result->GetValue(0, 1));
			Assert::AreEqual(3.75, result->GetValue(1, 0));
			Assert::AreEqual(5.5, result->GetValue(1, 1));
			Assert::AreEqual(5.75, result->GetValue(2, 0));
			Assert::AreEqual(8.5, result->GetValue(2, 1));
		}

		TEST_METHOD(Matrix_MultiplyDoubleCalled_ReturnsCorrectResult)
		{
			auto first = GetUniqueService<Matrix>();
			first->Set(3, 2, new double[] { 1, 2, 3, 4, 5, 6 });

			auto result = *first * 5;

			Assert::AreEqual(3, result->GetRows());
			Assert::AreEqual(2, result->GetCols());

			Assert::AreEqual(5.0, result->GetValue(0, 0));
			Assert::AreEqual(10.0, result->GetValue(0, 1));
			Assert::AreEqual(15.0, result->GetValue(1, 0));
			Assert::AreEqual(20.0, result->GetValue(1, 1));
			Assert::AreEqual(25.0, result->GetValue(2, 0));
			Assert::AreEqual(30.0, result->GetValue(2, 1));
		}
	};
}