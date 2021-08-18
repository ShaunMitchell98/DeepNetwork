#pragma once

#include "../Models/Matrix.h"
#include "ILogger.h"
#include <stdio.h>
#include <stdlib.h>

using namespace Models;

class Logger : public ILogger {
private:
	FILE* _fp;
	bool _enabled;
public:
	Logger();
	~Logger();
	void LogMatrix(Matrix* matrix);
	void LogMessage(const char* message...);
	void LogMessageWithoutDate(const char* message);
	void LogNumber(double number);
	void LogWhitespace();
	void LogNewline();
	void LogDoubleArray(double* array, int length);
	void LogLine(const char* message);
}; 
