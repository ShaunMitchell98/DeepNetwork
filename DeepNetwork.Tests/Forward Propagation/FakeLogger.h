#pragma once

#include "../DeepNetwork.Infrastructure/Logging/ILogger.h"

class FakeLogger : public ILogger
{
public:
	void LogMatrix(Matrix* matrix) {}
	void LogMessage(const char* message...) {}
	void LogMessageWithoutDate(const char* message) {}
	void LogNumber(double number) {}
	void LogWhitespace() {}
	void LogNewline() {}
	void LogDoubleArray(double* array, int length) {}
	void LogLine(const char* message) {}
};