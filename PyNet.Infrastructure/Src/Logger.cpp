#include <time.h>
#include "Logger.h"
#include <memory>

namespace PyNet::Infrastructure {

	Logger::Logger(bool log) {
		_enabled = log;

		if (_enabled) {
			_stream = std::ofstream("C:\\Users\\Shaun Mitchell\\Documents\\PyNet_Logs.txt");
		}
	}

	Logger::~Logger() {
		if (_enabled) {
			LogMessage("Closing logger...");
			_stream.close();
		}
	}

	void Logger::LogMessageWithoutDate(const char* message) {

		if (_enabled) {
			auto count = 0;
			while (message[count] != '\0') {
				count++;
			}

			_stream << message;
		}
	}

	void Logger::LogMessage(std::string message) {
		if (_enabled) {
			time_t _time = time(NULL);

			char buffer[26];
			ctime_s(buffer, 26, &_time);
			_stream << buffer;
			_stream << message;
		}
	}

	void Logger::LogWhitespace() {
		LogMessageWithoutDate(" ");
	}

	void Logger::LogNewline() {
		LogMessageWithoutDate("\n");
	}

	void Logger::LogNumber(double number) {
		_stream << number;
	}

	void Logger::LogLine(const char* message) {
		LogMessage(message);
		LogNewline();
	}
}
