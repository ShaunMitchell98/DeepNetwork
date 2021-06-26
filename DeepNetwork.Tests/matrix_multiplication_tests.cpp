#include "CppUnitTest.h"
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include "DeepNetwork.Infrastructure/matrix_multiplication.h"
#include "DeepNetwork.Infrastructure/dev_array.h"
#include "DeepNetwork.Infrastructure/matrix.h"
#include <math.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace matrix_multiplication_tests
{
	TEST_CLASS(MatrixMultiplicationTests)
	{
	public:
	
		TEST_METHOD(Function_GivenSquareMatrix_CalculatesMatrixMultiple)
		{
			float A[4] = { 1.0, 1.0, 1.0, 1.0 };
			float B[4] = { 1.0, 1.0, 1.0, 1.0 };
			float* C = (float*)malloc(4 * sizeof(float));

			matrix Am;
			Am.values = A;
			Am.rows = 2;
			Am.cols = 2;

			matrix Bm;
			Bm.values = B;
			Bm.rows = 2;
			Bm.cols = 2;

			matrix Cm;
			Cm.values = C;
			Cm.rows = Bm.cols;
			Cm.cols = Am.rows;

			matrixMultiply(Am, Bm, Cm);

            Assert::AreEqual((float)2, C[0]);
            Assert::AreEqual((float)2, C[1]);
            Assert::AreEqual((float)2, C[2]);
            Assert::AreEqual((float)2, C[3]);

			free(C);
		}

		TEST_METHOD(Function_GivenNonSquareMatrix_CalculatesMatrixMultiple)
		{
			float A[6] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
			float B[6] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
			float* C = (float*)malloc(4 * sizeof(float));

			matrix Am;
			Am.values = A;
			Am.rows = 2;
			Am.cols = 3;

			matrix Bm;
			Bm.values = B;
			Bm.rows = 3;
			Bm.cols = 2;

			matrix Cm;
			Cm.values = C;
			Cm.rows = Bm.cols;
			Cm.cols = Am.rows;

			matrixMultiply(Am, Bm, Cm);

			Assert::AreEqual((float)3, C[0]);
			Assert::AreEqual((float)3, C[1]);
			Assert::AreEqual((float)3, C[2]);
			Assert::AreEqual((float)3, C[3]);

			free(C);
		}

		TEST_METHOD(Function_GivenVectors_ReturnsResult)
		{
			float A[3] = { 1.0, 1.0, 1.0 };
			float B[3] = { 1.0, 1.0, 1.0 };
			float* C = (float*)malloc(1 * sizeof(float));

			matrix Am;
			Am.values = A;
			Am.rows = 1;
			Am.cols = 3;

			matrix Bm;
			Bm.values = B;
			Bm.rows = 3;
			Bm.cols = 1;

			matrix Cm;
			Cm.values = C;
			Cm.rows = Bm.cols;
			Cm.cols = Am.rows;

			matrixMultiply(Am, Bm, Cm);

			Assert::AreEqual((float)3, C[0]);

			free(C);
		}
	};
}
