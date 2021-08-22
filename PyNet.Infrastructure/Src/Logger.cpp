#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "Logger.h"
#include <memory>

Logger::Logger() {
	_enabled = true;
	_fp = NULL;

	if (_enabled) {
		fopen_s(&_fp, "C:\\Users\\Shaun Mitchell\\Documents\\PyNet_Logs.txt", "a");
	}
}

Logger::~Logger() {
	if (_enabled) {
		LogMessage("Closing logger...");
		fclose(_fp);
	}
}

void Logger::LogMatrix(Matrix* matrix) {

	if (_enabled) {
		for (auto row = 0; row < matrix->Rows; row++) {
			for (auto col = 0; col < matrix->Cols; col++) {
				LogNumber(matrix->GetValue(row, col));

				if (col != matrix->Cols - 1) {
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

	if (_enabled) {
		auto count = 0;
		while (message[count] != '\0') {
			count++;
		}

		fwrite(message, sizeof(char), count, _fp);
	}
}

void Logger::LogMessage(const char* message...) {
	if (_enabled) {
		auto t = time(NULL);
		struct tm tm;
		localtime_s(&tm, &t);
		fprintf(_fp, "%d-%02d-%02d %02d:%02d:%02d | ", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
		fprintf(_fp, message);
	}
}

void Logger::LogWhitespace() {
	LogMessageWithoutDate(" ");
}

void Logger::LogNewline() {
	LogMessageWithoutDate("\n");
}

void Logger::LogNumber(double number) {
	int size = 64;
	auto buffer = std::make_unique<char>(size);
	_gcvt_s(buffer.get(), size, number, sizeof(double));
	LogMessageWithoutDate(buffer.get());
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