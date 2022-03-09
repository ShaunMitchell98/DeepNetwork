export module PyNet.Infrastructure:Settings;

struct Settings {

	static auto factory() {
		return new Settings();
	}

	bool LoggingEnabled;
};