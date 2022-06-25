#pragma once

#include <string_view>
#include "PyNet.Models/Matrix.h"
#include <format>

namespace PyNet::Models {

	/// <summary>
	/// Base class for Loggers.
	/// </summary>
	class ILogger {

	public:
		virtual void LogDebugMatrix(const Matrix& matrix, format_args args = make_format_args(0)) const = 0;
		virtual void LogDebug(const string_view message, format_args args = make_format_args(0)) const = 0;
		virtual void LogInfo(const string_view message, format_args args = make_format_args(0)) const = 0;
		virtual void LogError(const string_view message, format_args args = make_format_args(0)) const = 0;
		virtual ~ILogger() = default;
	};
}
