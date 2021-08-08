#pragma once

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "logger.h"
#include <memory>

Logger::Logger() {
	fp = fopen("C:\\Users\\Shaun Mitchell\\Documents\\deep_network_logs.txt", "a");
	enabled = false;
}

Logger::~Logger() {
	LogMessage("Closing logger...");
	fclose(fp);
}
void Logger::DeleteLogFile() {

	FILE* fp;

	if (fp = fopen("C:\\Users\\Shaun Mitchell\\Documents\\deep_network_logs.txt", "r")) {
		fclose(fp);
		remove("C:\\Users\\Shaun Mitchell\\Documents\\deep_network_logs.txt");
	}
}

void Logger::LogMatrix(Matrix* matrix) {

	if (enabled) {
		for (auto i = 0; i < matrix->Rows; i++) {
			for (auto j = 0; j < matrix->Cols; j++) {
				int index = matrix->Cols * i + j;
				LogNumber(matrix->Values[index]);

				if (j != matrix->Cols - 1) {
					LogMessageWithoutDate(", ");
				}
			}

			LogNewline();
		}

		LogNewline();
		LogNewline();
		LogNewline();
		LogNewline();
	}
}
void Logger::LogMessageWithoutDate(const char* message) {

	if (enabled) {
		int count = 0;
		while (message[count] != '\0') {
			count++;
		}

		fwrite(message, sizeof(char), count, fp);
	}
}

void Logger::LogMessage(const char* message...) {
	if (enabled) {
		time_t t = time(NULL);
		struct tm tm = *localtime(&t);
		fprintf(fp, "%d-%02d-%02d %02d:%02d:%02d | ", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
		fprintf(fp, message);
	}
}

void Logger::LogWhitespace() {
	LogMessageWithoutDate(" ");
}

void Logger::LogNewline() {
	LogMessageWithoutDate("\n");
}

void Logger::LogNumber(double number) {
	auto buffer = std::make_unique<char>(sizeof(double));
	LogMessageWithoutDate(gcvt(number, sizeof(double), buffer.get()));
}

void Logger::LogDoubleArray(double* doubleArray, int length) {

	for (auto i = 0; i < length; i++) {
		LogNumber(doubleArray[i]);
		LogNewline();
	}
}

void Logger::LogLine(const char* message) {
	LogMessage(message);
	LogNewline();
}