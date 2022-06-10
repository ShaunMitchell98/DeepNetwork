#pragma once

namespace PyNet::Infrastructure
{
	struct TrainingState
	{
		static auto factory()
		{
			return new TrainingState();
		}

		int ExampleNumber;
		int Epoch;
		bool NewBatch;
	};
}