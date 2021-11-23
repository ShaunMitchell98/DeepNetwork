#pragma once

#include "PyNet.Models/Context.h"
#include "Logger.h"
#include "PyNet.Models.Cuda/CudaMatrix.h"
#include "PyNet.Models.Cuda/CudaVector.h"
#include "PyNet.Models.Cpu/CpuMatrix.h"
#include "PyNet.Models.Cpu/CpuVector.h"
#include "PyNet.Models.Cpu/CpuLogistic.h"

class UnitTest
{
public:

	di::Context* GetContext() {
		auto context = new di::ContextTmpl<PyNet::Infrastructure::Logger>();
		Settings* settings = new Settings();
		settings->LoggingEnabled = false;
		context->addInstance<Settings>(settings, true);
		context->addClass<CudaMatrix>(di::InstanceMode::Unique);
		context->addClass<CudaVector>(di::InstanceMode::Unique);
		context->addClass<PyNet::Models::Cpu::CpuLogistic>();
		return context;
	}
};

