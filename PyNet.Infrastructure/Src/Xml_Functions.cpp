#include "Xml_Functions.h"
#include <fstream>
#include "XmlReader.h"
#include "XmlWriter.h"
#include "PyNet.DI/Context.h"
#include "Intermediary.h"
#include "AdjustmentCalculator.h"
#include "PyNetwork.h"
#include "Headers.h"

using namespace PyNet::DI;
using namespace PyNet::Models;
using namespace std;

namespace PyNet::Infrastructure
{
	EXPORT void PyNetwork_Save(void* input, const char* filePath)
	{
		auto writer = XmlWriter::Create(filePath);
		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();
		auto pyNetwork = context->GetShared<PyNetwork>();

		writer->StartElement("Configuration");
		writer->StartElement("Layers");

		for (auto& layer : pyNetwork->Layers)
		{
			writer->StartElement("Layer");
			layer->Serialize(*writer);
			writer->EndElement();
		}

		writer->EndElement();

		writer->EndElement();
	}
}

