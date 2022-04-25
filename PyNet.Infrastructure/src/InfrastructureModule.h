#pragma once
#include <compare>
#include <memory>
#include "LayerPropagator.h"
#include "Settings.h"
#include "PyNet.Models.Cpu/CpuLogistic.h"
#include "PyNet.DI/Module.h"
#include "QuadraticLoss.h"
#include "LayerNormaliser.h"
#include "Logger.h"
#include "AdjustmentCalculator.h"
#include "SteepestDescent.h"
#include "GradientCalculator.h"
#include "PyNetwork.h"
#include "NetworkRunner.h"
#include "NetworkTrainer.h"
#include "BernoulliGenerator.h"
#include "DropoutRunner.h"

using namespace PyNet::Models;
using namespace PyNet::Models::Cpu;
using namespace PyNet::DI;
using namespace std;

namespace PyNet::Infrastructure {

	class InfrastructureModule : public Module {

	private:
		shared_ptr<Settings> _settings;

	public:

		InfrastructureModule(shared_ptr<Settings> settings) : _settings{ settings } {}

		void Load(ContextBuilder& builder) override {

			builder.RegisterType<CpuLogistic>()->As<Activation>();
				
			builder.RegisterType<QuadraticLoss>()->As<Loss>();
			
			builder.RegisterType<LayerNormaliser>()->AsSelf();

			builder.RegisterInstance<Settings>(_settings, InstanceMode::Shared);

			builder.RegisterType<Logger>()->As<ILogger>();
				
			builder.RegisterType<LayerPropagator>()->AsSelf();
				
			builder.RegisterType<AdjustmentCalculator>()->AsSelf();

			builder.RegisterType<SteepestDescent>()->As<TrainingAlgorithm>();

			builder.RegisterType<GradientCalculator>()->AsSelf();

			builder.RegisterType<PyNetwork>()->AsSelf();

			builder.RegisterType<NetworkRunner>()->AsSelf();

			builder.RegisterType<NetworkTrainer>()->AsSelf();

			builder.RegisterType<BernoulliGenerator>()->AsSelf();

			builder.RegisterType<DropoutRunner>()->AsSelf();
		}
	};
}