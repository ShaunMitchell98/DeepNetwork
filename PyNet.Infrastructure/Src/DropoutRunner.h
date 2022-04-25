#pragma once

#include "PyNet.Models/Vector.h"
#include "Settings.h"
#include "BernoulliGenerator.h"
#include <memory>

using namespace PyNet::Models;
using namespace std;

namespace PyNet::Infrastructure {

	/// <summary>
	/// Applies dropout to a vector.
	/// </summary>
	class DropoutRunner {
	private:

		shared_ptr<Settings> _settings;
		shared_ptr<BernoulliGenerator> _bernoulliGenerator;
		DropoutRunner(shared_ptr<Settings> settings, shared_ptr<BernoulliGenerator> bernoulliGenerator) : _settings(settings), _bernoulliGenerator(bernoulliGenerator) {}

	public:

		auto static factory(shared_ptr<Settings> settings, shared_ptr<BernoulliGenerator> bernoulliGenerator) {
			return new DropoutRunner(settings, bernoulliGenerator);
		}

		/// <summary>
		/// Applies dropout to the given vector.
		/// </summary>
		/// <param name="input">A vector</param>
		/// <returns>A dropped vector</returns>
		void ApplyDropout(Vector& input) const;
	};

}
