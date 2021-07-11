#pragma once

#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

void DeleteLogFile() {

	FILE* fp;

	if (fp = fopen("C:\\Users\\Shaun Mitchell\\Documents\\matrix_multiple.txt", "r")) {
		fclose(fp);
		remove("C:\\Users\\Shaun Mitchell\\Documents\\matrix_multiple.txt");
	}
}

void LogMatrix(matrix matrix) {
	FILE* fp = fopen("C:\\Users\\Shaun Mitchell\\Documents\\matrix_multiple.txt", "a");

	for (int i = 0; i < matrix.rows; i++) {
		for (int j = 0; j < matrix.cols; j++) {
			char buffer[100];
			gcvt(matrix.values[matrix.cols * i + j], 20, buffer);
			fwrite(buffer, sizeof(char), 20, fp);
			fwrite(" ", sizeof(char), 1, fp);
		}

		fwrite("\n", sizeof(char), 1, fp);
	}

	fwrite("\n", sizeof(char), 1, fp);
	fwrite("\n", sizeof(char), 1, fp);
	fwrite("\n", sizeof(char), 1, fp);
	fwrite("\n", sizeof(char), 1, fp);

	fclose(fp);
}

void LogMessage(char* message) {
	FILE* fp = fopen("C:\\Users\\Shaun Mitchell\\Documents\\matrix_multiple.txt", "a");

	int length = 0;
	while (message[length] != '\0') {
		length++;
	}

	fwrite(message, sizeof(char), length, fp);
	fclose(fp);
}

void LogNumber(float number) {
	char buffer[sizeof(float)];
	LogMessage(gcvt(number, sizeof(float), buffer));
}

void LogWhitespace() {
	LogMessage(" ");
}

void LogNewline() {
	LogMessage("\n");
}

void LogFloatArray(float* array, int length) {
	for (int i = 0; i < length; i++) {
		LogNumber(array[i]);
		LogWhitespace();
	}
	LogNewline();
}

void LogLine(char* message) {
	LogMessage(message);
	LogNewline();
}