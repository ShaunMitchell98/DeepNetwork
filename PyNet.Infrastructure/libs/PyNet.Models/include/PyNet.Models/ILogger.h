#pragma once

#include "Matrix.h" 

using namespace PyNet::Models;

class ILogger {

public:
	virtual void LogMatrix(Matrix* matrix) = 0;
	virtual void LogMessage(const char* message...) = 0;
	virtual void LogMessageWithoutDate(const char* message) = 0;
	virtual void LogNumber(double number) = 0;
	virtual void LogWhitespace() = 0;
	virtual void LogNewline() = 0;
	virtual void LogVector(std::vector<double> values) = 0;
	virtual void LogLine(const char* message) = 0;
};
