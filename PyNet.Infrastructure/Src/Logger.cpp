#include <time.h>
#include "Logger.h"
#include <memory>
#include <chrono>

namespace PyNet::Infrastructure {

	void Logger::LogMessageWithoutDate(std::string_view message) const {

		if (_enabled) {
			auto stream = std::ofstream(_fileName, std::ios_base::app);
			stream << message;
			stream.close();
		}
	}

	void Logger::LogMessage(std::string_view message) const {
		if (_enabled) {

			auto time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
			auto stream = std::ofstream(_fileName, std::ios_base::app);
			stream << std::ctime(&time);
			stream << message;
			stream.close();
		}
	}

	void Logger::LogLine(std::string_view message) const {
		LogMessage(message);
		LogMessageWithoutDate("\n");
	}

	void Logger::LogMatrix(const Matrix& m) const {
		if (_enabled) {
			LogMessage(m.ToString());
		}
	}
}
