#include "matrix_multiplication.h"

void matrix_multiply(Matrix* A, Matrix* B, Matrix* C) {

    for (auto i = 0; i < A->Rows; i++) {
        for (auto j = 0; j < B->Cols; j++) {
            double tempValue = 0;
            for (auto k = 0; k < A->Cols; k++) {
                tempValue += A->GetValue(i, k) * B->GetValue(k, j);
            }

            C->SetValue(j, i, tempValue);
        }
    }
}

void vector_add(Vector* A, Vector* B, Vector* C) {
    for (auto i = 0; i < A->Rows; i++) {
        C->SetValue(i, A->GetValue(i) + B->GetValue(i));
    }
}