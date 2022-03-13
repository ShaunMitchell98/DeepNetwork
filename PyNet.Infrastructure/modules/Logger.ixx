module;
#include <fstream>
#include <time.h>
#include <memory>
#include <chrono>
export module PyNet.Infrastructure:Logger;

import :Settings;
import PyNet.Models;

using namespace PyNet::Models;
using namespace std;

class __declspec(dllexport) Logger : public ILogger {
private:
	bool _enabled;
	const char* _fileName = "PyNet_Logs.txt";
	Logger(bool log) : _enabled{ log } {};
public:

	static auto factory(shared_ptr<Settings> settings) {
		return new Logger{ settings->LoggingEnabled };
	}

	void LogMessage(string_view message) override {
		if (_enabled) {

			auto time = chrono::system_clock::to_time_t(chrono::system_clock::now());
			auto stream = ofstream(_fileName, ios_base::app);
			stream << ctime(&time);
			stream << message;
			stream.close();
		}
	}

	void LogMessageWithoutDate(string_view message) override {

		if (_enabled) {
			auto stream = ofstream(_fileName, ios_base::app);
			stream << message;
			stream.close();
		}
	}

	void LogLine(string_view message) override {
		LogMessage(message);
		LogMessageWithoutDate("\n");
	}
};

