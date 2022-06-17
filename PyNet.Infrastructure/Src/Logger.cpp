#include <time.h>
#include "Logger.h"
#include <memory>

#define EXTRA_LOGGING false

namespace PyNet::Infrastructure {

	void Logger::LogMessageWithoutPreamble(std::string_view message) const {

		if (_settings->LoggingEnabled) {
			auto stream = std::ofstream(_fileName, std::ios_base::app);
			stream << message;
			stream.close();
		}
	}

	void Logger::LogMessage(const string_view message, format_args args) const
	{
		if (_settings->LoggingEnabled)
		{

			if (_trainingState->ExampleNumber % 10 == 0 || EXTRA_LOGGING)
			{
				auto stream = std::ofstream(_fileName, std::ios_base::app);

				stream << std::fixed;
				auto temp = vformat(message, args);

				auto output = format("Epoch: {}, Example Number: {}, {}", _trainingState->Epoch, _settings->StartExampleNumber + _trainingState->ExampleNumber, temp);
				stream << output;
				stream.close();
			}

		}
	}

	void Logger::LogLine(const string_view message, format_args args) const
	{

		if (_trainingState->ExampleNumber % 10 == 0 || EXTRA_LOGGING)
		{
			LogMessage(message, args);
			LogMessageWithoutPreamble("\n");
		}
	}

	void Logger::LogMatrix(const Matrix& m) const
	{
		if (_settings->LoggingEnabled && _trainingState->ExampleNumber % 10 == 0 || EXTRA_LOGGING)
		{
			LogMessageWithoutPreamble(m.ToString());
		}
	}
}
