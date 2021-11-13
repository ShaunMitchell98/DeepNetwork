#pragma once

struct Settings {

	static auto factory() {
		return new Settings();
	}

	bool LoggingEnabled;
	bool CudaEnabled;
};