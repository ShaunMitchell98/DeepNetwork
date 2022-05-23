#include <gtest/gtest.h>
#include <memory>
#include "PyNet.Models.Cpu/CpuMatrix.h"
#include <cmath>

using namespace std;

namespace PyNet::Models::Cpu::Tests {

	TEST(CpuMatrixTests, CpuMatrix_MultiplyWithMatrixCalled_ReturnsProduct) {
		auto matrix1 = make_unique<CpuMatrix>();

		matrix1->Initialise(800, 500, false);

		auto firstMatrixElement = 2.98;

		for (auto& element : *matrix1) {
			element = firstMatrixElement;
		}

		auto matrix2 = make_unique<CpuMatrix>();
		matrix2->Initialise(500, 300, false);

		auto secondMatrixElement = 9.21;

		for (auto& element : *matrix2) {
			element = secondMatrixElement;
		}

		auto result = *matrix1 * *matrix2;

		for (auto& element : *result) {
 			ASSERT_EQ((int)(500.0 * firstMatrixElement * secondMatrixElement), (int)element);
		}
	}

	TEST(CpuMatrixTests, CpuMatrix_AddWithMatrixCalled_ReturnsSum) {
		auto matrix1 = make_unique<CpuMatrix>();

		matrix1->Initialise(800, 500, false);

		auto firstMatrixElement = 7.43;

		for (auto& element : *matrix1) {
			element = firstMatrixElement;
		}

		auto matrix2 = make_unique<CpuMatrix>();
		matrix2->Initialise(800, 500, false);

		auto secondMatrixElement = 1.92;

		for (auto& element : *matrix2) {
			element = secondMatrixElement;
		}

		auto result = *matrix1 + *matrix2;

		for (auto& element : *result) {
			ASSERT_EQ(firstMatrixElement + secondMatrixElement, element);
		}
	}

	TEST(CpuMatrixTests, CpuMatrix_AddWithDoubleCalled_ReturnsSum) {
		auto matrix1 = make_unique<CpuMatrix>();

		matrix1->Initialise(800, 500, false);

		auto firstMatrixElement = 2.31;

		for (auto& element : *matrix1) {
			element = firstMatrixElement;
		}

		auto d = 3.98;

		auto result = *matrix1 + d;

		for (auto& element : *result) {
			ASSERT_EQ(firstMatrixElement + d, element);
		}
	}

	TEST(CpuMatrixTests, CpuMatrix_MultiplyWithDoubleCalled_ReturnsProduct) {
		auto matrix1 = make_unique<CpuMatrix>();

		matrix1->Initialise(800, 500, false);

		auto firstMatrixElement = 2.31;

		for (auto& element : *matrix1) {
			element = firstMatrixElement;
		}

		auto d = 3.98;

		auto result = *matrix1 * d;

		for (auto& element : *result) {
			ASSERT_EQ(firstMatrixElement * d, element);
		}
	}

	TEST(CpuMatrixTests, CpuMatrix_SubtractWithMatrixCalled_ReturnsDifference) {
		auto matrix1 = make_unique<CpuMatrix>();

		matrix1->Initialise(800, 500, false);

		auto firstMatrixElement = 7.43;

		for (auto& element : *matrix1) {
			element = firstMatrixElement;
		}

		auto matrix2 = make_unique<CpuMatrix>();
		matrix2->Initialise(800, 500, false);

		auto secondMatrixElement = 1.92;

		for (auto& element : *matrix2) {
			element = secondMatrixElement;
		}

		auto result = *matrix1 - *matrix2;

		for (auto& element : *result) {
			ASSERT_EQ(firstMatrixElement - secondMatrixElement, element);
		}
	}

	TEST(CpuMatrixTests, CpuMatrix_MinusCalledOnMatrix_ReturnsNegative) {
		auto matrix1 = make_unique<CpuMatrix>();

		matrix1->Initialise(800, 500, false);

		auto firstMatrixElement = 7.43;

		for (auto& element : *matrix1) {
			element = firstMatrixElement;
		}

		auto result = -*matrix1;

		for (auto& element : *result) {
			ASSERT_EQ(-firstMatrixElement, element);
		}
	}

	TEST(CpuMatrixTests, CpuMatrix_PlusEqualsCalledOnMatrix_ReturnsResult) {
		auto matrix1 = make_unique<CpuMatrix>();

		matrix1->Initialise(800, 500, false);

		auto firstMatrixElement = 7.43;

		for (auto& element : *matrix1) {
			element = firstMatrixElement;
		}

		auto matrix2 = make_unique<CpuMatrix>();
		matrix2->Initialise(800, 500, false);

		auto secondMatrixElement = 1.92;

		for (auto& element : *matrix2) {
			element = secondMatrixElement;
		}

		*matrix1 += *matrix2;

		for (auto& element : *matrix1) {
			ASSERT_EQ(firstMatrixElement + secondMatrixElement, element);
		}
	}

	TEST(CpuMatrixTests, CpuMatrix_CopyCalled_ReturnsCopy) {
		auto matrix1 = make_unique<CpuMatrix>();

		matrix1->Initialise(800, 500, false);

		auto result = matrix1->Copy();

		ASSERT_EQ(800, matrix1->GetRows());
		ASSERT_EQ(500, matrix1->GetCols());
	}

	TEST(CpuMatrixTests, CpuMatrix_ExpCalled_ReturnsResult) {
		auto matrix1 = make_unique<CpuMatrix>();

		matrix1->Initialise(800, 500, false);

		auto firstMatrixElement = 7.43;

		for (auto& element : *matrix1) {
			element = firstMatrixElement;
		}

		auto result = matrix1->Exp();

		auto expectedValue = exp(firstMatrixElement);

		for (auto& element : *result) {
			ASSERT_EQ(expectedValue, element);
		}
	}

	TEST(CpuMatrixTests, CpuMatrix_ReciprocalCalled_ReturnsResult) {
		auto matrix1 = make_unique<CpuMatrix>();

		matrix1->Initialise(800, 500, false);

		auto firstMatrixElement = 7.43;

		for (auto& element : *matrix1) {
			element = firstMatrixElement;
		}

		auto result = matrix1->Reciprocal();

		auto expectedValue = 1 / firstMatrixElement;

		for (auto& element : *result) {
			ASSERT_EQ(expectedValue, element);
		}
	}

	TEST(CpuMatrixTests, CpuMatrix_MaxCalledWithSmallerNumber_ReturnsElement) {
		auto matrix1 = make_unique<CpuMatrix>();

		matrix1->Initialise(800, 500, false);

		auto firstMatrixElement = 7.43;

		for (auto& element : *matrix1) {
			element = firstMatrixElement;
		}

		auto result = matrix1->Max(3.2);

		for (auto& element : *result) {
			ASSERT_EQ(firstMatrixElement, element);
		}
	}

	TEST(CpuMatrixTests, CpuMatrix_MaxCalledWithLargerNumber_ReturnsNumber) {
		auto matrix1 = make_unique<CpuMatrix>();

		matrix1->Initialise(800, 500, false);

		auto firstMatrixElement = 7.43;

		for (auto& element : *matrix1) {
			element = firstMatrixElement;
		}

		auto result = matrix1->Max(12.3);

		for (auto& element : *result) {
			ASSERT_EQ(12.3, element);
		}
	}

	TEST(CpuMatrixTests, CpuMatrix_StepCalledWithElementLessThanZero_ReturnsZero) {
		auto matrix1 = make_unique<CpuMatrix>();

		matrix1->Initialise(800, 500, false);

		auto firstMatrixElement = -7.43;

		for (auto& element : *matrix1) {
			element = firstMatrixElement;
		}

		auto result = matrix1->Step();

		for (auto& element : *result) {
			ASSERT_EQ(0, element);
		}
	}

	TEST(CpuMatrixTests, CpuMatrix_StepCalledWithElementEqualToZero_ReturnsZero) {
		auto matrix1 = make_unique<CpuMatrix>();

		matrix1->Initialise(800, 500, false);

		auto firstMatrixElement = 0.0;

		for (auto& element : *matrix1) {
			element = firstMatrixElement;
		}

		auto result = matrix1->Step();

		for (auto& element : *result) {
			ASSERT_EQ(0, element);
		}
	}

	TEST(CpuMatrixTests, CpuMatrix_StepCalledWithElementGreaterThanZero_ReturnsOne) {
		auto matrix1 = make_unique<CpuMatrix>();

		matrix1->Initialise(800, 500, false);

		auto firstMatrixElement = 4.13;

		for (auto& element : *matrix1) {
			element = firstMatrixElement;
		}

		auto result = matrix1->Step();

		for (auto& element : *result) {
			ASSERT_EQ(1, element);
		}
	}

	TEST(CpuMatrixTests, CpuMatrix_HadamardProductCalled_ReturnsProduct) {
		auto matrix1 = make_unique<CpuMatrix>();

		matrix1->Initialise(800, 500, false);

		auto firstMatrixElement = 7.43;

		for (auto& element : *matrix1) {
			element = firstMatrixElement;
		}

		auto matrix2 = make_unique<CpuMatrix>();
		matrix2->Initialise(800, 500, false);

		auto secondMatrixElement = 9.87;

		for (auto& element : *matrix2) {
			element = secondMatrixElement;
		}

		auto result = *matrix1 ^ *matrix2;

		for (auto& element : *result) {
			ASSERT_EQ(firstMatrixElement * secondMatrixElement, element);
		}
	}
}
