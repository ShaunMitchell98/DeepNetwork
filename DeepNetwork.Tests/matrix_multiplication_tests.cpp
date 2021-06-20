#include "pch.h"
#include "CppUnitTest.h"
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include "DeepNetwork/matrix_multiplication.h"
#include "DeepNetwork/dev_array.h"
#include <math.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace matrix_multiplication_tests
{
	TEST_CLASS(MatrixMultiplicationTests)
	{
	public:
	
		TEST_METHOD(Function_GivenSquareMatrix_CalculatesMatrixMultiple)
		{
            int N = 2;
            int size = N * N;
            std::vector<float> h_A(size, 1);
            std::vector<float> h_B(size, 1);
            std::vector<float> h_C(size);

            dev_array<float> d_A(size);
            dev_array<float> d_B(size);
            dev_array<float> d_C(size);

            d_A.set(&h_A[0], size);
            d_B.set(&h_B[0], size);

            matrixMultiply(d_A.getData(), d_B.getData(), d_C.getData(), N);
            d_C.get(&h_C[0], size);

            Assert::AreEqual((float)2, h_C[0]);
            Assert::AreEqual((float)2, h_C[1]);
            Assert::AreEqual((float)2, h_C[2]);
            Assert::AreEqual((float)2, h_C[3]);
		}
	};
}
