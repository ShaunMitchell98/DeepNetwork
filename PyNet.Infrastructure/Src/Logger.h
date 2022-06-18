#pragma once

#include "PyNet.Models/ILogger.h"
#include "Settings.h"
#include "TrainingState.h"
#include "Headers.h"
#include <fstream>
#include <filesystem>

using namespace std;
using namespace std::filesystem;
using namespace PyNet::Models;

namespace PyNet::Infrastructure {

	class EXPORT Logger : public ILogger {
	private:
		const char* _fileName = "PyNet_Logs.txt";
		shared_ptr<Settings> _settings;
		shared_ptr<TrainingState> _trainingState;
		Logger(shared_ptr<Settings> settings, shared_ptr<TrainingState> trainingState) : _settings{ settings }, _trainingState{ trainingState } 
		{
			remove(_fileName);
		};

		/// <summary>
		/// Generates a log message prepended with the current epoch and example number.
		/// </summary>
		/// <param name="message">The message to be logged.</param>
		void LogMessage(const string_view message, format_args args = make_format_args()) const;
		void LogMessageWithoutPreamble(std::string_view message) const;
		void LogInternal(const string_view message, format_args args, LogLevel logLevel) const;
	public:

		typedef ILogger base;

		static auto factory(shared_ptr<Settings> settings, shared_ptr<TrainingState> trainingState) {
			return new Logger(settings, trainingState);
		}

		void LogDebugMatrix(const Matrix& matrix, format_args args = make_format_args(0)) const override;
		void LogDebug(const string_view message, format_args args = make_format_args(0)) const override;
		void LogInfo(const string_view message, format_args args = make_format_args(0)) const override;
		void LogError(const string_view message, format_args args = make_format_args(0)) const override;
		~Logger() override = default;
	};
}

