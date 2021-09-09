#include "CppUnitTest.h"
#include "PyNetwork_Functions.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;

namespace PyNetInfrastructureFunctionalTests
{
	TEST_CLASS(TrainingTests)
	{
	public:
		
		TEST_METHOD(PyNet_GivenData_DoesTraining)
		{
			auto network = PyNetwork_New(784, true);
			PyNetwork_AddLayer(network, 500, ActivationFunctionType::Logistic);
			PyNetwork_AddLayer(network, 129, ActivationFunctionType::Logistic);
			PyNetwork_AddLayer(network, 10, ActivationFunctionType::Logistic);

			string folderPath = "C:\\Users\\Shaun Mitchell\\source\\repos\\PyNet\\PyNet.Infrastructure.FunctionalTests\\Examples\\Mnist_Fashion\\";
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
			PyNetwork_Train(network, inputs.data(), labels.data(), 10, 5, 0.01);

			for (auto i = 0; i < inputs.size(); i++) {
				delete[] inputs[i];
				delete[] labels[i];
			}
		}

		void GetData(string folderPath, string fileName, vector<double*> inputs) {

			auto inputValues = vector<vector<double>>();

			for (auto i = 0; i < 10; i++) {

				inputValues.push_back(vector<double>());
				string filePath = folderPath + fileName + to_string(i) + ".txt";

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

		void GetLabels(string folderPath, string fileName, vector<double*> labels) {

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
	};
}
