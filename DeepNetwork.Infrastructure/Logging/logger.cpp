#pragma once

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "logger.h"


Logger::Logger() {
	fp = fopen("C:\\Users\\Shaun Mitchell\\Documents\\deep_network_logs.txt", "a");
}

Logger::~Logger() {
	fclose(fp);
}
void Logger::DeleteLogFile() {

	FILE* fp;

	if (fp = fopen("C:\\Users\\Shaun Mitchell\\Documents\\deep_network_logs.txt", "r")) {
		fclose(fp);
		remove("C:\\Users\\Shaun Mitchell\\Documents\\matrix_multiple.txt");
	}
}

void Logger::LogMatrix(matrix matrix) {

	for (int i = 0; i < matrix.rows; i++) {
		for (int j = 0; j < matrix.cols; j++) {
			char buffer[100];
			gcvt(matrix.values[matrix.cols * i + j], 20, buffer);
			fwrite(buffer, sizeof(char), 20, fp);

			if (j != matrix.cols - 1) {
				fwrite(", ", sizeof(char), 2, fp);
			}
		}

		fwrite("\n", sizeof(char), 1, fp);
	}

	fwrite("\n", sizeof(char), 1, fp);
	fwrite("\n", sizeof(char), 1, fp);
	fwrite("\n", sizeof(char), 1, fp);
	fwrite("\n", sizeof(char), 1, fp);
}

void Logger::LogMessageWithoutDate(const char* message...) {
	fprintf(fp, message);
}

void Logger::LogMessage(const char* message...) {
	time_t t = time(NULL);
	struct tm tm = *localtime(&t);
	fprintf(fp, "%d-%02d-%02d %02d:%02d:%02d | ", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
	fprintf(fp, message);
}

void Logger::LogWhitespace() {
	LogMessageWithoutDate(" ");
}

void Logger::LogNewline() {
	LogMessageWithoutDate("\n");
}

void Logger::LogNumber(float number) {
	char buffer[sizeof(float)];
	LogMessageWithoutDate(gcvt(number, sizeof(float), buffer));
}

void Logger::LogFloatArray(float* array, int length) {
	for (int i = 0; i < length; i++) {
		LogNumber(array[i]);
		LogWhitespace();
	}
	LogNewline();
}

void Logger::LogLine(const char* message...) {
	LogMessage(message);
	LogNewline();
}