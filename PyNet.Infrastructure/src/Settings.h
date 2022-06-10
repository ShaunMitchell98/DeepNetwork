#pragma once

#include "RunMode.h"

/// <summary>
/// A container for settings.
/// </summary>
struct Settings {

	static auto factory() {
		return new Settings();
	}

	/// <summary>
	/// Determines whether logging messages should be outputted.
	/// </summary>
	bool LoggingEnabled = false;
	bool CudaEnabled = false;
	RunMode RunMode = RunMode::Training;
	double BaseLearningRate = 0.0;
	int BatchSize = 0;
	int Epochs = 0;
	int NumberOfExamples = 0;
	int StartExampleNumber = 0;
	double Momentum = 0;
};