#pragma once
#include <string>
#include <fstream>
#include <memory>
#include <stack>

using namespace std;

namespace PyNet::Infrastructure {
	class XmlReader {
	private:
		ifstream _stream;
		stack<string> _openElements;
	public:
		XmlReader(string_view filePath);
		static unique_ptr<XmlReader> Create(string_view filePath);
		bool FindNode(string_view value);
		string ReadContent();
	};
};