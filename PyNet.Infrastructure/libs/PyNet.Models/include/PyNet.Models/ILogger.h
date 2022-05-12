#pragma once

#include <string_view>
#include "PyNet.Models/Matrix.h"

namespace PyNet::Models {

	/// <summary>
	/// Base class for Loggers.
	/// </summary>
	class ILogger {
	public:

		/// <summary>
		/// Generates a log message prepended with the current date and time.
		/// </summary>
		/// <param name="message">The message to be logged.</param>
		virtual void LogMessage(std::string_view message) const = 0;
		virtual void LogMessageWithoutDate(std::string_view message) const = 0;
		virtual void LogLine(std::string_view message) const = 0;
		virtual ~ILogger() = default;
		virtual void LogMatrix(const PyNet::Models::Matrix& m) const = 0;
	};
}
