#pragma once

struct VariableLearningSettings
{
	double ErrorThreshold = 0;
	double LRDecrease = 0;
	double LRIncrease = 0;

	static auto factory() 
	{
		return new VariableLearningSettings();
	}
};