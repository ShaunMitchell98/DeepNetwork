#include "BernoulliGenerator.h"
#include <random>

namespace PyNet::Infrastructure {

	unique_ptr<Matrix> BernoulliGenerator::GetBernoulliVector(const Matrix& input, double probability) const {

		default_random_engine gen;
		bernoulli_distribution dist(probability);

		auto output = input.Copy();

		for (auto& m : *output) {
			if (dist(gen)) {
				m = 1;
			}
			else {
				m = 0;
			}
		}

		return output;
	}
}
