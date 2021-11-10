#pragma once

#include <string>

class ILogger {

public:
	virtual void LogMessage(std::string message) = 0;
	virtual void LogMessageWithoutDate(const char* message) = 0;
	virtual void LogNumber(double number) = 0;
	virtual void LogWhitespace() = 0;
	virtual void LogNewline() = 0;
	virtual void LogLine(const char* message) = 0;
};
