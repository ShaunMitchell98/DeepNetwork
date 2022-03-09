module;
#include <string>
#include <fstream>
#include <memory>
#include <stack>
export module PyNet.Infrastructure:XmlWriter;

using namespace std;

class XmlWriter {
private:
	ofstream _stream;
	stack<string_view> _openElements;
public:
	XmlWriter(string_view filePath) { _stream = ofstream(filePath.data(), ios_base::app); }

	static unique_ptr<XmlWriter> Create(string_view filePath) {
		auto network = make_unique<XmlWriter>(filePath);
		network->WriteDeclaration();
		return move(network);
	}

	void WriteDeclaration() {
		_stream << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
	}

	void StartElement(string_view name) {
		_stream << "<";
		_stream << name;
		_stream << ">\n";
		_openElements.push(name);
	}

	void WriteString(string_view value) {
		_stream << value;
	}

	void EndElement() {
		auto& name = _openElements.top();
		_stream << "</";
		_stream << name;
		_stream << ">\n";
		_openElements.pop();
	}
};
