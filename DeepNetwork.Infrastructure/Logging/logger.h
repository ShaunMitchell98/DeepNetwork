#pragma once

#include "../matrix.h"
#include <stdio.h>
#include <stdlib.h>

class Logger {
private:
	FILE* fp;
public:
	Logger();
	~Logger();
	void DeleteLogFile();
	void LogMatrix(matrix matrix);
	void LogMessage(const char* message...);
	void LogMessageWithoutDate(const char* message...);
	void LogNumber(float number);
	void LogWhitespace();
	void LogNewline();
	void LogFloatArray(float* array, int length);
	void LogLine(const char* message...);
};
