#include "BernoulliGenerator.h"
#include <random>

namespace PyNet::Infrastructure {

	unique_ptr<Vector> BernoulliGenerator::GetBernoulliVector(const Vector& input) const {

		default_random_engine gen;
		bernoulli_distribution dist(input.GetDropoutRate());

		auto output = input.Copy();

		for (auto i = 0; i < input.GetRows(); i++) {
			if (dist(gen)) {
				(*output)[i] = 1;
			}
			else {
				(*output)[i] = 0;
			}
		}

		return move(output);
	}
}
