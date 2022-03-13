module;
#include <compare>
export module PyNet.Infrastructure:InfrastructureModule;

import PyNet.Models.Cpu;
import PyNet.Models;
import :LayerPropagator;
import :LayerNormaliser;
import :AdjustmentCalculator;
import :SteepestDescent;
import :GradientCalculator;
import :QuadraticLoss;
import :Settings;
import :Logger;
import :NetworkTrainer;
import :NetworkRunner;
import :PyNetwork;
import PyNet.DI;

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
