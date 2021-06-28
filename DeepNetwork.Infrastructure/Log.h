#pragma once

#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

void LogMatrix(matrix matrix) {
	FILE* fp = fopen("C:\\Users\\Shaun Mitchell\\Documents\\matrix_multiple.txt", "a");

	for (int i = 0; i < (matrix.rows * matrix.cols);  i++) {
		char buffer[100];
		gcvt(matrix.values[i], 5, buffer);
		fwrite(buffer, sizeof(char), 1, fp);
		fwrite("\n", sizeof(char), 1, fp);
	}

	fclose(fp);
}