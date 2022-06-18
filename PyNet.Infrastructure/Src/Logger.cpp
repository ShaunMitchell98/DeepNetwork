#include <time.h>
#include "Logger.h"
#include <memory>

namespace PyNet::Infrastructure 
{
	void Logger::LogDebugMatrix(const Matrix& matrix, format_args args) const
	{
		if (_settings->LogLevel <= LogLevel::DEBUG) 
		{
			LogMessage(matrix.ToString(), args);
			LogMessageWithoutPreamble("\n");
		}
	}

	void Logger::LogDebug(const string_view message, format_args args) const
	{
		LogInternal(message, args, LogLevel::DEBUG);
	}


	void Logger::LogInfo(const string_view message, format_args args) const
	{
		LogInternal(message, args, LogLevel::INFO, 10);
	}

	void Logger::LogError(const string_view message, format_args args) const
	{
		LogInternal(message, args, LogLevel::ERROR);
	}

	void Logger::LogInternal(const string_view message, format_args args, LogLevel logLevel, int logInterval) const 
	{
		if (_settings->LogLevel <= logLevel && _trainingState->ExampleNumber % logInterval == 0)
		{
			LogMessage(message, args);
			LogMessageWithoutPreamble("\n");
		}
	}

	void Logger::LogMessageWithoutPreamble(string_view message) const
	{
		auto stream = ofstream(_fileName, ios_base::app);
		stream << message;
		stream.close();
	}

	void Logger::LogMessage(const string_view message, format_args args) const
	{
		auto stream = ofstream(_fileName, ios_base::app);

		stream << fixed;
		auto temp = vformat(message, args);

		auto output = format("Epoch: {}, Example Number: {}, {}", _trainingState->Epoch, _settings->StartExampleNumber + _trainingState->ExampleNumber, temp);
		stream << output;
		stream.close();
	}
}