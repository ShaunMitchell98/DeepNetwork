#pragma once

#include "PyNet.Models/ILogger.h"
#include "Settings.h"
#include "Headers.h"
#include <fstream>

using namespace std;
using namespace PyNet::Models;

namespace PyNet::Infrastructure {

	class EXPORT Logger : public ILogger {
	private:
		bool _enabled;
		const char* _fileName = "PyNet_Logs.txt";
		Logger(bool log) : _enabled{ log } {};
	public:

		typedef ILogger base;

		static auto factory(shared_ptr<Settings> settings) {
			return new Logger{ settings->LoggingEnabled };
		}

		void LogMessage(string_view message) const override;
		void LogMessageWithoutDate(string_view message) const override;
		void LogLine(string_view message) const override;
		void LogMatrix(const Matrix& m) const override;
		~Logger() override = default;
	};
}

