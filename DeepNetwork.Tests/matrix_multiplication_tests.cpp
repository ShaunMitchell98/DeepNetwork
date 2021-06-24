#include "CppUnitTest.h"
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include "DeepNetwork.Infrastructure/matrix_multiplication.h"
#include "DeepNetwork.Infrastructure/dev_array.h"
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

			matrixMultiply(A, B, C, 2);

            Assert::AreEqual((float)2, C[0]);
            Assert::AreEqual((float)2, C[1]);
            Assert::AreEqual((float)2, C[2]);
            Assert::AreEqual((float)2, C[3]);

			free(C);
		}
	};
}
