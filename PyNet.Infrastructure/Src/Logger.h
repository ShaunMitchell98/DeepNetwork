#pragma once

#include "PyNet.Models/Matrix.h"
#include "ILogger.h"
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

class Logger : public ILogger {
private:
	bool _enabled;
	std::ofstream _stream;
public:
	Logger(bool log);
	~Logger();
	void LogMatrix(PyNet::Models::Matrix* matrix);
	void LogMessage(const char* message...);
	void LogMessageWithoutDate(const char* message);
	void LogNumber(double number);
	void LogWhitespace();
	void LogNewline();
	void LogVector(std::vector<double> values);
	void LogLine(const char* message);
}; 
