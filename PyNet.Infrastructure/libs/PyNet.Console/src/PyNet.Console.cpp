#include "PyNetwork_Functions.h"
#include "Xml_Functions.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "Activations/Activation.h"
#include "Settings.h"

using namespace std;
using namespace PyNet::Infrastructure;
using namespace PyNet::Infrastructure::Activations;

void GetData(string folderPath, string fileName, vector<double*>& inputs) {

	auto inputValues = vector<vector<double>>();

	for (auto i = 0; i < 10; i++) {

		inputValues.push_back(vector<double>());
		auto filePath = folderPath + fileName + to_string(i) + ".txt";

		ifstream ifs(filePath, ifstream::in);

		if (!ifs.bad()) {
			auto pBuf = ifs.rdbuf();
			do {
				string inputValue = "";
				char ch = pBuf->sgetc();

				while (ch != '\n') {
					inputValue += ch;
					pBuf->snextc();
					ch = pBuf->sgetc();
				}

				inputValues[i].push_back(atof(inputValue.c_str()));
			} while (pBuf->snextc() != EOF);
		}

		ifs.close();
	}

	for (auto j = 0; j < inputValues.size(); j++) {
		for (auto k = 0; k < inputValues[j].size(); k++) {
			inputs[j][k] = inputValues[j][k];
		}
	}
}

void GetLabels(string folderPath, string fileName, vector<double*>& labels) {

	auto labelValues = vector<int>();

	string filePath = folderPath + fileName + ".txt";

	ifstream ifs(filePath);

	if (!ifs.bad()) {
		auto pBuf = ifs.rdbuf();
		do {
			string inputValue = "";
			char ch = pBuf->sgetc();

			while (ch != '\n') {
				inputValue += ch;
				pBuf->snextc();
				ch = pBuf->sgetc();
			}

			labelValues.push_back(atoi(inputValue.c_str()));
		} while (pBuf->snextc() != EOF);
	}

	ifs.close();

	for (auto i = 0; i < labelValues.size(); i++) {
		for (auto j = 0; j < 10; j++) {
			if (labelValues[i] == j) {
				labels[i][j] = 1;
			}
			else {
				labels[i][j] = 0;
			}
		}
	}
}

int main()
{
	auto intermediary = PyNetwork_Initialise(LogLevel::INFO, false);
	PyNetwork_AddInputLayer(intermediary, 784, 1, 1.0);
	PyNetwork_AddDenseLayer(intermediary, 500, ActivationFunctionType::Logistic, 0.8);
	PyNetwork_AddDenseLayer(intermediary, 129, ActivationFunctionType::Logistic, 0.8);
	PyNetwork_AddDenseLayer(intermediary, 10, ActivationFunctionType::Logistic, 0.8);
	//PyNetwork_AddSoftmaxLayer(intermediary);
	//PyNetwork_SetVariableLearning(intermediary, 0.04, 0.7, 1.05);


#ifdef _WIN32
	string folderPath = "C:\\Users\\Shaun Mitchell\\source\\repos\\PyNet\\PyNet.Infrastructure\\tests\\Resources\\";
#else
	string folderPath = "/Users/shaunmitchell/pynet/PyNet.Infrastructure/tests/resources/";
#endif
	string trainingExamplesFileName = "Training_Example";
	string trainingLabelsFileName = "Training_Labels";


	vector<double*> inputs;
	vector<double*> labels;

	for (auto i = 0; i < 10; i++) {
		inputs.push_back(new double[784]);
		labels.push_back(new double[10]);
	}

	GetData(folderPath, trainingExamplesFileName, inputs);
	GetLabels(folderPath, trainingLabelsFileName, labels);

	auto settings = make_unique<Settings>();
	settings->BaseLearningRate = 0.01;
	settings->BatchSize = 1;
	settings->Epochs = 1;
	settings->Momentum = 0.7;
	settings->NumberOfExamples = 10;
	settings->StartExampleNumber = 0;

	PyNetwork_Train(intermediary, inputs.data(), labels.data(), settings.get());
	PyNetwork_Run(intermediary, inputs[0]);
	PyNetwork_Save(intermediary, "C:\\Users\\Shaun Mitchell\\source\\repos\\PyNet\\Network.xml");
	PyNetwork_Destruct(intermediary);
}
	