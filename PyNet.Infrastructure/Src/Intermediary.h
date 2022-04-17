#pragma once
#include <memory>
#include "PyNet.DI/Context.h"
#include "Settings.h"

using namespace PyNet::DI;
using namespace std;

class Intermediary {
private:
	shared_ptr<Context> _context;
	shared_ptr<Settings> _settings;

public:
	Intermediary(shared_ptr<Context> context, shared_ptr<Settings> settings) : _context{ context }, _settings{ settings } {}

	shared_ptr<Context> GetContext() {
		return _context;
	}
};