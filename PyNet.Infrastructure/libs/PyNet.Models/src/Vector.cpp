#include "Vector.h"

namespace PyNet::Models {

	double Vector::operator|(const Vector& v) const {
		if (v.GetRows() != GetRows()) {
			throw "Cannot calculate dot product for vectors with different lengths";
		}

		double result = 0;

		for (auto i = 0; i < v.GetRows(); i++) {
			result += (*this)[i] * v[i];
		}

		return result;
	}
}

