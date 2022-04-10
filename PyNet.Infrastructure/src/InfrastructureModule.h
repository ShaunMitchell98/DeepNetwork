#pragma once

#include "PyNet.Models.Cpu/CpuLogistic.h"
#include "PyNet.DI/Module.h"
#include "PyNet.DI/ContextBuilder.h"
#include "QuadraticLoss.h"
#include "LayerNormaliser.h"
#include "Settings.h"
#include "Logger.h"
#include "LayerPropagator.h"
#include "AdjustmentCalculator.h"
#include "SteepestDescent.h"
#include "GradientCalculator.h"
#include "PyNetwork.h"
#include "NetworkRunner.h"
#include "NetworkTrainer.h"

using namespace PyNet::Models;
using namespace PyNet::Models::Cpu;
using namespace PyNet::DI;

namespace PyNet::Infrastructure {

	class InfrastructureModule : public Module {

	private:
		const bool _logEnabled;

	public:

		InfrastructureModule(bool logEnabled) : _logEnabled{ logEnabled } {}

		void Load(ContextBuilder& builder) override {

			builder.RegisterType<CpuLogistic>().As<Activation>();
				
			builder.RegisterType<QuadraticLoss>().As<Loss>();
			
			builder.RegisterType<LayerNormaliser>().AsSelf();

			builder.AddInstance<Settings>(new Settings{ _logEnabled }, InstanceMode::Shared);

			builder.RegisterType<Logger>().AsSelf().As<ILogger>();
				
			builder.RegisterType<LayerPropagator>().AsSelf();
				
			builder.RegisterType<AdjustmentCalculator>().AsSelf();

			builder.RegisterType<SteepestDescent>().As<TrainingAlgorithm>();

			builder.RegisterType<GradientCalculator>().AsSelf();

			builder.RegisterType<PyNetwork>().AsSelf();

			builder.RegisterType<NetworkRunner>().AsSelf();

			builder.RegisterType<NetworkTrainer>().AsSelf();
		}
	};
}