export module PyNet.Infrastructure:VariableLearningSettings;

export struct VariableLearningSettings {
	double ErrorThreshold = 0;
	double LRDecrease = 0;
	double LRIncrease = 0;
};