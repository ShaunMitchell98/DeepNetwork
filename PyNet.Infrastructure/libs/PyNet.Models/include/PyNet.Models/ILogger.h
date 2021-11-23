#pragma once

#include <string_view>

namespace PyNet::Models {

	class ILogger {
	public:
		virtual void LogMessage(std::string_view message) = 0;
		virtual void LogMessageWithoutDate(std::string_view message) = 0;
		virtual void LogLine(std::string_view message) = 0;
	};
}
