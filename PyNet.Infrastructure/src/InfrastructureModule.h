#pragma once

#include "PyNet.DI/Module.h"
#include "PyNet.Models.Cpu/CpuLogistic.h"
#include "QuadraticLoss.h"
#include "Logger.h"
#include "Settings.h"

namespace PyNet::Infrastructure {

	class InfrastructureModule : public PyNet::DI::Module {

	private:
		const bool _logEnabled;

	public:

		InfrastructureModule(bool logEnabled) : _logEnabled{ logEnabled } {}

		void Load(PyNet::DI::ContextBuilder* builder) override {

			builder
				->AddClass<PyNet::Models::Cpu::CpuLogistic>()
				->AddClass<QuadraticLoss>()
				->AddClass<Logger>()
				->AddInstance<Settings>(new Settings {_logEnabled }, PyNet::DI::InstanceMode::Shared);
		}
	};
}
