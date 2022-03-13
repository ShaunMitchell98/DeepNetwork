module;
#include <memory>
#include <fstream>
export module PyNet.Infrastructure:XmlFunctions;

import :Setup;
import :PyNetwork;
import :AdjustmentCalculator;
import :VariableLearningSettings;
import :XmlReader;
import :XmlWriter;
import PyNet.Models;
import PyNet.DI;

using namespace PyNet::DI;
using namespace PyNet::Models;
using namespace std;

export namespace PyNet::Infrastructure {
	extern "C" {

		__declspec(dllexport) int PyNetwork_Load(void* input, const char* filePath) {

			auto reader = XmlReader::Create(filePath);
			auto context = static_cast<Context*>(input);
			auto pyNetwork = context->GetShared<PyNetwork>();
			auto adjustmentCalculator = context->GetShared<AdjustmentCalculator>();

			if (reader->FindNode("Configuration")) {
				if (reader->FindNode("Weights")) {
					while (reader->FindNode("Weight")) {
						auto weightMatrix = context->GetUnique<Matrix>();
						weightMatrix->Load(reader->ReadContent());
						adjustmentCalculator->AddMatrix(weightMatrix->GetRows(), weightMatrix->GetCols());
						pyNetwork->Weights.push_back(move(weightMatrix));
					}
				}

				if (reader->FindNode("Biases")) {
					while (reader->FindNode("Bias")) {
						auto biasVector = context->GetUnique<Vector>();
						biasVector->Load(reader->ReadContent());
						pyNetwork->Biases.push_back(move(biasVector));
					}
				}

				auto layer = context->GetUnique<Vector>();
				layer->Initialise(pyNetwork->Weights[0]->GetCols(), false);
				pyNetwork->Layers.push_back(move(layer));

				for (auto& m : pyNetwork->Weights) {
					auto layer = context->GetUnique<Vector>();
					layer->Initialise(m->GetRows(), false);
					pyNetwork->Layers.push_back(move(layer));
				}
			}

			return pyNetwork->GetLastLayer().GetRows();
		}

		__declspec(dllexport) void PyNetwork_Save(void* input, const char* filePath) {

			auto writer = XmlWriter::Create(filePath);
			auto context = static_cast<Context*>(input);
			auto pyNetwork = context->GetShared<PyNetwork>();

			writer->StartElement("Configuration");
			writer->StartElement("Weights");
			for (auto i = 0; i < pyNetwork->Weights.size(); i++) {
				writer->StartElement("Weight");
				writer->WriteString(pyNetwork->Weights[i]->ToString());
				writer->EndElement();
			}

			writer->EndElement();

			writer->StartElement("Biases");
			for (auto i = 0; i < pyNetwork->Biases.size(); i++) {
				writer->StartElement("Bias");
				writer->WriteString(pyNetwork->Biases[i]->ToString());
				writer->EndElement();
			}

			writer->EndElement();

			writer->EndElement();
		}
	}
}