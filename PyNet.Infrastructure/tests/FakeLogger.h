#pragma once

#include "../Src/ILogger.h"

class FakeLogger : public ILogger
{
public:
	void LogMatrix(Matrix* matrix) {}
	void LogMessage(const char* message...) {}
	void LogMessageWithoutDate(const char* message) {}
	void LogNumber(double number) {}
	void LogWhitespace() {}
	void LogNewline() {}
	void LogVector(std::vector<double> values) {}
	void LogLine(const char* message) {}
};