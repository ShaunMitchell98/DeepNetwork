#pragma once

#include <vector>

using namespace std;

void matrix_multiply(const vector<double>& A, const vector<double>& B, vector<double>& C, int Arows, int Acols, int Bcols);
void multiply_matrix_and_double(const vector<double>& A, const double B, vector<double>& C, int Arows, int Acols);
void add_matrix_and_double(const vector<double>& A, const double B, vector<double>& C, int Arows, int Acols);
void matrix_add(const vector<double>& A, const vector<double>& B, vector<double>& C, int Arows, int Acols);
void matrix_subtract(const vector<double>& A, const vector<double>& B, vector<double>& C, int Arows, int Acols);
void matrix_addition_assignment(vector<double>& A, const vector<double>& B, int Arows, int Acols);
void matrix_exp(const vector<double> A, vector<double>& B, int Arows, int Acols);
void matrix_reciprocal(const vector<double>& A, vector<double>& B, int Arows, int Acols);
void matrix_max(const vector<double>& A, double input, vector<double>& B, int Arows, int Acols);
void matrix_step(const vector<double> A, vector<double>& B, int Arows, int Acols);
void matrix_hadamard(const vector<double>& A, const vector<double>& B, vector<double>& C, int Arows, int Acols);