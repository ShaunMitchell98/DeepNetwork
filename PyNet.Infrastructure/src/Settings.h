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
	bool LoggingEnabled;
	RunMode RunMode;
};