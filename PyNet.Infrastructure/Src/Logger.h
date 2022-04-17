#pragma once

#include "PyNet.Models/ILogger.h"
#include "Settings.h"
#include <fstream>

namespace PyNet::Infrastructure {

	class Logger : public PyNet::Models::ILogger {
	private:
		bool _enabled;
		const char* _fileName = "PyNet_Logs.txt";
		Logger(bool log) : _enabled{ log } {};
	public:

		typedef ILogger base;

		static auto factory(std::shared_ptr<Settings> settings) {
			return new Logger{ settings->LoggingEnabled };
		}

		void LogMessage(std::string_view message) override;
		void LogMessageWithoutDate(std::string_view message) override;
		void LogLine(std::string_view message) override;
		~Logger() override = default;
	};
}

