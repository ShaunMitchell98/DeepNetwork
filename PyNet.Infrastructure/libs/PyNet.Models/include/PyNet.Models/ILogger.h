#pragma once

#include <string_view>
#include "PyNet.Models/Matrix.h"
#include <format>

namespace PyNet::Models
{

	/// <summary>
	/// Base class for Loggers.
	/// </summary>
	class ILogger
	{
		public:

		/// <summary>
		/// Generates a log message prepended with the current epoch and example number.
		/// </summary>
		/// <param name="message">The message to be logged.</param>
		virtual void LogMessage(const string_view message, format_args args = make_format_args()) const = 0;
		virtual void LogMessageWithoutPreamble(std::string_view message) const = 0;
		virtual void LogLine(const string_view message, format_args args = make_format_args(0)) const = 0;
		virtual ~ILogger() = default;
		virtual void LogMatrix(const PyNet::Models::Matrix& m) const = 0;
	};
}
