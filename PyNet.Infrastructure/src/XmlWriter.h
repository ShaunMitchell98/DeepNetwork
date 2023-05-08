#pragma once

#include <string>
#include <fstream>
#include <memory>
#include <stack>

using namespace std;

namespace PyNet::Infrastructure {

	class XmlWriter {
	private:
		ofstream _stream;
		stack<string_view> _openElements;
	public:
		XmlWriter(string_view filePath);
		static unique_ptr<XmlWriter> Create(string_view filePath);
		void WriteDeclaration();
		void StartElement(string_view name);
		void WriteString(string_view value);
		void EndElement();
	};
}
