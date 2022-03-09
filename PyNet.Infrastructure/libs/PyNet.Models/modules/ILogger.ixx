module;
#include <string_view>
export module PyNet.Models:ILogger;

using namespace std;

export namespace PyNet::Models {

	class ILogger {
	public:
		virtual void LogMessage(string_view message) = 0;
		virtual void LogMessageWithoutDate(string_view message) = 0;
		virtual void LogLine(string_view message) = 0;
	};
}
