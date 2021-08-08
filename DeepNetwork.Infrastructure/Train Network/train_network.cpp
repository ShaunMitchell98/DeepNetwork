#include "train_network.h"
#include "../NetworkTrainer/NetworkTrainer.h"

double train_network(PyNetwork* network, Matrix* expectedLayer) {

    auto networkTrainer = std::make_unique<NetworkTrainer>(network);
    return networkTrainer->TrainNetwork(network, expectedLayer);
}