#include "train_network.h"
#include "../NetworkTrainer/NetworkTrainer.h"

double train_network(network network, matrix expectedLayer) {

    auto networkTrainer = std::make_unique<NetworkTrainer>(network);
    return networkTrainer->TrainNetwork(network, expectedLayer);
}