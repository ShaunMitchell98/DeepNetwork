#include "../Include/PyNetwork_Functions.h"

#include "PyNetwork.h"
#include "Logger.h"
#include "PyNet.Models.Cuda/CudaMatrix.h"
#include "PyNet.Models.Cpu/CpuMatrix.h"
#include <PyNet.Models.Cpu/ActivationProvider.h>
#include <PyNet.Models/Context.h>
#include "Settings.h"

namespace PyNet::Infrastructure {
	extern "C" {

		void AddMatrix(di::Context* context, bool cudaEnabled) {
			if (cudaEnabled) {
				context->addClass<CudaMatrix>();
			}
			else
			{
				context->addClass<CpuMatrix>();
				context->addFactory(PyNet::Models::Cpu::factory);
			}
		}

		di::Context* GetContext(bool cudaEnabled, bool log) {
			auto context = new di::Context();
			context->addClass<Settings>();
			
			AddMatrix(context, cudaEnabled);
			auto settings = new Settings();
			settings->LoggingEnabled = log;
			context->addInstance<Settings>(settings, true);
			return context;
		}

		void* PyNetwork_New(int count, bool log, bool cudaEnabled) {
			auto context = GetContext(cudaEnabled, log);
			auto pyNetwork = new PyNetwork(context->get<PyNetwork>());
			pyNetwork->AddInitialLayer(count);
			return pyNetwork;
		}

		void PyNetwork_AddLayer(void* pyNetwork, int count, PyNet::Models::ActivationFunctionType activationFunctionType) {
			((PyNetwork*)pyNetwork)->AddLayer(count, activationFunctionType);
		}

		double* PyNetwork_Run(void* pyNetwork, double* input_layer) {
			return ((PyNetwork*)pyNetwork)->Run(input_layer);
		}

		void PyNetwork_Train(void* pyNetwork, double** inputLayers, double** expectedOutputs, int numberOfExamples, int batchSize, double learningRate) {
			((PyNetwork*)pyNetwork)->Train(inputLayers, expectedOutputs, numberOfExamples, batchSize, learningRate);
		}
	}
}

