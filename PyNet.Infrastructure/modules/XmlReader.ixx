module;
#include <string>
#include <fstream>
#include <memory>
#include <stack>
export module PyNet.Infrastructure:XmlReader;

using namespace std;

class XmlReader {
private:
	ifstream _stream;
	stack<string> _openElements;
public:
	XmlReader(string_view filePath) {
		_stream = ifstream(filePath.data(), ios::in);
	}

	static unique_ptr<XmlReader> Create(string_view filePath) { return make_unique<XmlReader>(filePath); }

	bool FindNode(string_view value) {
		if (_stream.eof()) {
			return false;
		}

		auto targetValue = string("<") + value.data() + ">";
		string endValue;

		if (!_openElements.empty()) {
			endValue = "</" + _openElements.top() + ">";
		}

		string currentValue;

		bool condition = true;

		while (condition) {
			getline(_stream, currentValue);

			if (currentValue == endValue) {
				_openElements.pop();
				return false;
			}

			if (currentValue == targetValue) {
				condition = false;
			}
		}

		_openElements.push(string(value));
		return true;
	}

	string ReadContent() {
		string content;
		string line;

		bool condition = true;
		auto& current = _openElements.top();
		auto targetValue = "</" + current + ">";

		while (condition) {
			getline(_stream, line);

			if (targetValue != line) {
				content += line;
				content += '\n';
			}
			else {
				condition = false;
			}
		}

		_openElements.pop();

		return content;
	}
};