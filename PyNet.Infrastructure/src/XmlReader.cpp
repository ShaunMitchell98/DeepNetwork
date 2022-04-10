#include "XmlReader.h"
#include <iostream>

namespace PyNet::Infrastructure {

	XmlReader::XmlReader(string_view filePath) {
		_stream = ifstream(filePath.data(), ios::in);
	}

	std::unique_ptr<XmlReader> XmlReader::Create(string_view filePath) {
		return make_unique<XmlReader>(filePath);
	}

	bool XmlReader::FindNode(string_view value) {

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

		_openElements.push(std::string(value));
		return true;
	}

	string XmlReader::ReadContent() {

		string content;
		string line;

		bool condition = true;
		auto current = _openElements.top();
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
}
