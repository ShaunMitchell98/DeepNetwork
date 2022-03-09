export module PyNet.DI:Module;

import :ContextBuilder;

export namespace PyNet::DI {

	class Module {
	public:
		virtual void Load(ContextBuilder& builder) = 0;
	};
}
