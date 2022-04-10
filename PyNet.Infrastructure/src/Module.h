//#pragma once
//
//#include "di.h"
//#include "Logger.h"
//#include "Settings.h"
//
//namespace PyNet::Infrastructure {
//	class Module : public cpp_di::Module {
//	public:
//		void Load() {
//			container->Register<Logger>()
//				.As<ILogger>()
//				.GlobalSingleton();
//
//			container->Register<Settings>([](cpp_di::Container& scope) {
//				auto settings = new Settings();
//				settings->LoggingEnabled = true;
//				return *settings;
//				}, false)
//				.As<Settings>()
//				.InstancePerScope();
//		}
//	};
//}