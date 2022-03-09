module;
#include <compare>
export module PyNet.Infrastructure:InfrastructureModule;

import PyNet.Models.Cpu;
import :LayerPropagator;
import :AdjustmentCalculator;
import :SteepestDescent;
import :GradientCalculator;
import :QuadraticLoss;
import :Settings;
import :Logger;
import :PyNetwork;
import PyNet.DI;

using namespace PyNet::Models::Cpu;

namespace PyNet::Infrastructure {

	class InfrastructureModule : public PyNet::DI::Module {

	private:
		const bool _logEnabled;

	public:

		InfrastructureModule(bool logEnabled) : _logEnabled{ logEnabled } {}

		void Load(PyNet::DI::ContextBuilder& builder) override {

			builder
				.RegisterType<CpuLogistic>()
				.RegisterType<QuadraticLoss>()
				.AddInstance<Settings>(new Settings{ _logEnabled }, PyNet::DI::InstanceMode::Shared)
				.RegisterType<Logger>()
				.RegisterType<LayerPropagator>()
				.RegisterType<AdjustmentCalculator>()
				.RegisterType<SteepestDescent>()
				.RegisterType<GradientCalculator>()
				.RegisterType<PyNetwork>(PyNet::DI::InstanceMode::Unique);
		}
	};

}
