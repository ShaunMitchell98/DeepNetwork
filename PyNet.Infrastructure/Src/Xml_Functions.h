#pragma once

#include "Headers.h"

namespace PyNet::Infrastructure {
	extern "C" {

		EXPORT int PyNetwork_Load(void* input, const char* filePath) {

		/*	auto reader = XmlReader::Create(filePath);
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

			return pyNetwork->GetOutputLayer().GetRows();*/
			return 0;
		}

		EXPORT void PyNetwork_Save(void* input, const char* filePath);
	}
}