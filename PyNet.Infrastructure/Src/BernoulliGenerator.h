#pragma once

#include "PyNet.Models/Matrix.h"
#include <ranges>
#include "Headers.h"
#include <memory>

using namespace std;
using namespace PyNet::Models;

namespace PyNet::Infrastructure {

	/// <summary>
	/// Generates a vector using the Bernoulli distribution.
	/// </summary>
	class EXPORT BernoulliGenerator{
	private:

	public:

		auto static factory() {
			return new BernoulliGenerator();
		}

		unique_ptr<Matrix> GetBernoulliVector(const Matrix& input, double probability) const;
	};
}