export module PyNet.Infrastructure:Settings;

struct Settings {

	__declspec(dllexport) static auto factory() {
		return new Settings();
	}

	bool LoggingEnabled;
};