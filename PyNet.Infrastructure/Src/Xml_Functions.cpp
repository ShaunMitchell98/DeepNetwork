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
	EXPORT int PyNetwork_Load(void* input, const char* filePath)
	{
		auto reader = XmlReader::Create(filePath);
		auto intermediary = static_cast<Intermediary*>(input);
		auto context = intermediary->GetContext();
		auto pyNetwork = context->GetShared<PyNetwork>();

		if (reader->FindNode("Configuration"))
		{
			if (reader->FindNode("Layers"))
			{
				while (reader->FindNode("Layer"))
				{
					if (reader->FindNode("Type"))
					{
						auto typeName = reader->ReadContent();
						auto layer = context->GetUnique<Layer>(typeName);
						layer->Deserialize(*reader);
						pyNetwork->Layers.push_back(move(layer));
						reader->PopNode();
					}
				}
			}
		}

		return pyNetwork->Layers.back()->GetRows();
	}

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

