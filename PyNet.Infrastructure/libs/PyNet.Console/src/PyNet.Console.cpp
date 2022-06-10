#include "PyNetwork_Functions.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "PyNet.Models/Activation.h"

using namespace std;
using namespace PyNet::Infrastructure;

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
	auto intermediary = PyNetwork_Initialise(true, false);
	PyNetwork_AddLayer(intermediary, 784, PyNet::Models::ActivationFunctionType::Logistic, 0.8);
	PyNetwork_AddLayer(intermediary, 500, PyNet::Models::ActivationFunctionType::Logistic, 0.5);
	PyNetwork_AddLayer(intermediary, 129, PyNet::Models::ActivationFunctionType::Logistic, 0.5);
	PyNetwork_AddLayer(intermediary, 10, PyNet::Models::ActivationFunctionType::Logistic, 0.5);
	PyNetwork_SetVariableLearning(intermediary, 0.04, 0.7, 1.05);


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
	PyNetwork_Train(intermediary, inputs.data(), labels.data(), 10, 5, 0.01, 0, 10, 0);
	PyNetwork_Run(intermediary, inputs[0]);
	PyNetwork_Destruct(intermediary);

}
	