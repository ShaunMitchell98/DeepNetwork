#pragma once

#include "PyNet.Models/Matrix.h"
#include "PyNet.Models/ILogger.h"
#include "Settings.h"
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

namespace PyNet::Infrastructure {

	class Logger : public PyNet::Models::ILogger {
	private:
		bool _enabled;
		std::ofstream _stream;
	public:

		typedef ILogger base;

		static auto factory(Settings& settings) {
			return new Logger{ settings.LoggingEnabled };
		}

		Logger(bool log);
		~Logger();
		void LogMessage(std::string message);
		void LogMessageWithoutDate(const char* message);
		void LogNumber(double number);
		void LogWhitespace();
		void LogNewline();
		void LogLine(const char* message);
	};
}

