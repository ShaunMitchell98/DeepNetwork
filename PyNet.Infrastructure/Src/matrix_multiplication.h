#pragma once

#include "PyNet.Models/Matrix.h"
#include "PyNet.Models/Vector.h"

using namespace Models;

void matrix_multiply(Matrix* A, Matrix* B, Matrix* C);
void vector_add(Vector* A, Vector* B, Vector* C);
