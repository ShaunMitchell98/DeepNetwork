//#include "../Src/LayerPropagator.h"
//#include "FakeLogger.h"
//#include <memory>
//#include <gtest/gtest.h>
//
//namespace PyNet::Infrastructure::Tests
//{
//	TEST(LayerPropagatorTests, Propagator_GivenWeightsAndInput_ReturnsOutput) {
//		double weights[4] = { 1, 2, 3, 4 };
//		auto weightMatrix = std::make_unique<PyNet::Models::Matrix>(2, 2, weights, true);
//
//		double inputLayer[2] = { 1, 1 };
//		auto inputLayerVector = std::make_unique<PyNet::Models::Vector>(2, inputLayer, ActivationFunctions::ActivationFunctionType::Logistic, true);
//
//		auto outputLayerVector = std::make_unique<PyNet::Models::Vector>(2, ActivationFunctions::ActivationFunctionType::Logistic, true);
//		auto biasesVector = std::make_unique<PyNet::Models::Vector>(2, true);
//
//		auto layerPropagator = std::make_unique<LayerPropagator>(std::make_shared<FakeLogger>());
//		layerPropagator->PropagateLayer(weightMatrix.get(), inputLayerVector.get(), biasesVector.get(), outputLayerVector.get());
//
//		EXPECT_EQ(0.95257412682243336, outputLayerVector->GetValue(0));
//		EXPECT_EQ(0.99908894880559940, outputLayerVector->GetValue(1));
//	}
//}