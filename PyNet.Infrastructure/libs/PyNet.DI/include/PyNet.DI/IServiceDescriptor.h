#pragma once
#include <memory>
#include <any>
#include <functional>
#include "IServiceProvider.h"
#include <string>

 using namespace std;

 namespace PyNet::DI 
 {
	 class IServiceDescriptor
	 {
		 public:
		 virtual void Reset() = 0;
		 virtual void MakeReferenceWeak() = 0;

		 shared_ptr<any> ImplementationInstance;

		 virtual const string& GetServiceType() const = 0;
		 virtual const function<shared_ptr<void>(IServiceProvider&)>& GetImplementationFactory() const = 0;
	 };
 }

