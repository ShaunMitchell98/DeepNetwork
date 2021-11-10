#include "NetworkTrainer.h"
#include "PyNet.Models/Logistic.h"

NetworkTrainer::NetworkTrainer(ILogger* logger, AdjustmentCalculator* adjustmentCalculator, Settings* settings) {
    _logger = logger;
    _adjustmentCalculator = adjustmentCalculator;
    _settings = settings;
    dError_dActivatedLayerAbove = new Vector(0);
    dError_dActivatedOutput = new Vector(0);
}

double NetworkTrainer::TrainNetwork(std::vector<Matrix*> weightMatrices, std::vector<Vector*> layers, PyNet::Models::Vector* expectedLayer) {

    double error = CalculateErrorDerivativeForFinalLayer(layers[layers.size() - 1], expectedLayer);
    GetAdjustments(weightMatrices, layers);
    _adjustmentCalculator->SetNewBatch(false);

    _logger->LogMessage("I am returning: ");
    _logger->LogNumber(error);
    _logger->LogNewline();

    return error;
}

double NetworkTrainer::CalculateErrorDerivativeForFinalLayer(PyNet::Models::Vector* finalLayer, PyNet::Models::Vector* expectedLayer) {

    //dError_dActivatedOutput.clear();
    //_logger->LogMessage("Expected layer is:");
    //_logger->LogVector(expectedLayer->Values);
    double error = 0;
    for (int b = 0; b < finalLayer->GetRows(); b++) {
        error += 0.5 * (expectedLayer->GetValue(b) - finalLayer->GetValue(b)) * (expectedLayer->GetValue(b) - finalLayer->GetValue(b));
        _logger->LogMessage("Temp error is ");
        _logger->LogNumber(error);
        _logger->LogNewline();
    }

    *dError_dActivatedOutput = *finalLayer - *expectedLayer;
    _logger->LogLine("Calculated derivatives for final layer.");
    _logger->LogMessage("Error is: ");
    _logger->LogNumber(error);
    _logger->LogNewline();

    return error;
}

void NetworkTrainer::GetdError_dActivatedOutput(Matrix* weightMatrix, PyNet::Models::Vector* inputLayer, PyNet::Models::Vector* outputLayer) {
    //dError_dOutputCurrent.clear();
    _logger->LogLine("Calculating error derivative with respect to current output layer.");
    _logger->LogNumber(weightMatrix->GetCols());
    _logger->LogNewline();

    auto dActivatedLayerAbove_dOutput = std::make_unique<Vector>(outputLayer->GetRows());
    outputLayer->CalculateActivationDerivative(dActivatedLayerAbove_dOutput.get());

    auto dError_dOutput = std::make_unique<Vector>(outputLayer->GetRows());
    dError_dOutput.reset(&(*dError_dActivatedLayerAbove ^ *dActivatedLayerAbove_dOutput.get()));

    auto weightMatrixTranspose = std::make_unique<Matrix>(weightMatrix->GetCols(), weightMatrix->GetRows());
    weightMatrixTranspose.reset(~*weightMatrix);
    //*dError_dActivatedInput = (*dError_dOutput_transpose) * *weightMatrix;
    *dError_dActivatedOutput = *weightMatrixTranspose * *dError_dOutput;

    _logger->LogMessage("dError_dOutputCurrent: ");
    //_logger->LogVector(dError_dOutputCurrent);
}

void NetworkTrainer::UpdateWeights(std::vector<Matrix*> weightMatrices, std::vector<Vector*> biases, double learningRate) {
    _logger->LogLine("Updating weights...");
    for (int weightMatrixIndex = static_cast<int>(weightMatrices.size() - 1); weightMatrixIndex >= 0; weightMatrixIndex--) {
        Matrix* weightMatrix = weightMatrices[weightMatrixIndex];
        Vector* bias = biases[weightMatrixIndex];

        *bias = *bias - (*_adjustmentCalculator->GetBiasAdjustment(weightMatrixIndex) * learningRate);
        *weightMatrix = *weightMatrix - (*_adjustmentCalculator->GetWeightAdjustment(weightMatrixIndex) * learningRate);
    }

    _adjustmentCalculator->SetNewBatch(true);
}

void NetworkTrainer::GetAdjustmentsForWeightMatrix(Matrix* weightMatrix, Vector* inputLayer, Vector* outputLayer, int weightMatrixIndex) {
    _logger->LogMessage("Calcuating loop for weight matrix: ");
    _logger->LogNumber(weightMatrixIndex);
    _logger->LogNewline();

    _logger->LogLine("Calculating adjustments.");

    auto temp_name = std::make_unique<Vector>(outputLayer->GetRows());
    outputLayer->CalculateActivationDerivative(temp_name.get());

    auto dError_dBias = 0.0;
    dError_dBias = *dError_dActivatedOutput | *temp_name;
    _adjustmentCalculator->AddBiasAdjustment(weightMatrixIndex, dError_dBias);

    /// /////////////////////////////////

    auto dActivatedOutput_dOutput = std::make_unique<Vector>(outputLayer->GetRows());
    outputLayer->CalculateActivationDerivative(dActivatedOutput_dOutput.get());

    auto dError_dOutput = std::make_unique<Matrix>(outputLayer->GetRows(), 1);
    *dError_dOutput = *dError_dActivatedOutput ^ *dActivatedOutput_dOutput;

    auto dOutput_dWeight = inputLayer;

    auto dOutput_dWeight_Transpose = std::make_unique<Matrix>(1, inputLayer->GetRows());
    dOutput_dWeight_Transpose.reset(~*dOutput_dWeight);
    auto dError_dWeight = std::make_unique<Matrix>(outputLayer->GetRows(), outputLayer->GetRows());
    *dError_dWeight = *dError_dOutput * *dOutput_dWeight_Transpose;

    _adjustmentCalculator->AddWeightAdjustment(weightMatrixIndex, dError_dWeight.get());

    dError_dActivatedLayerAbove = dError_dActivatedOutput;
    GetdError_dActivatedOutput(weightMatrix, inputLayer, outputLayer);
}

void NetworkTrainer::GetAdjustments(std::vector<Matrix*> weightMatrices, std::vector<Vector*> layers) {
    for (int a = static_cast<int>(weightMatrices.size() - 1); a >= 0; a--) {
        GetAdjustmentsForWeightMatrix(weightMatrices[a], layers[a], layers[(size_t)a+1], a);
    }

    _logger->LogLine("End of GetAdjustments");
}