#pragma once

#include "PyNet.Models/Vector.h"
#include "PyNet.DI/Context.h"
#include "Settings.h"
#include <memory>

using namespace std;
using namespace PyNet::DI;
using namespace PyNet::Models;

namespace PyNet::Infrastructure {

	/// <summary>
	/// Generates a vector using the Bernoulli distribution.
	/// </summary>
	class BernoulliGenerator{
	private:
		shared_ptr<Context> _context;
		shared_ptr<Settings> _settings;

		BernoulliGenerator(shared_ptr<Context> context, shared_ptr<Settings> settings) : _context(context), _settings(settings) {}
	public:

		auto static factory(shared_ptr<Context> context, shared_ptr<Settings> settings) {
			return new BernoulliGenerator(context, settings);
		}

		unique_ptr<Vector> GetBernoulliVector(const Vector& input) const;
	};
}