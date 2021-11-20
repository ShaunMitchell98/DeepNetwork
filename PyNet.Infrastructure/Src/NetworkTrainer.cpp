#include "NetworkTrainer.h"

double NetworkTrainer::TrainNetwork(std::vector<Matrix*> weightMatrices, std::vector<Vector*> layers, PyNet::Models::Vector* expectedLayer) {

    double error = CalculateErrorDerivativeForFinalLayer(layers[layers.size() - 1], expectedLayer);
    GetAdjustments(weightMatrices, layers);
    _adjustmentCalculator.SetNewBatch(false);

    _logger.LogMessage("I am returning: ");
    _logger.LogNumber(error);
    _logger.LogNewline();

    return error;
}

double NetworkTrainer::CalculateErrorDerivativeForFinalLayer(PyNet::Models::Vector* finalLayer, PyNet::Models::Vector* expectedLayer) {

    //dError_dActivatedOutput.clear();
    //_logger.LogMessage("Expected layer is:");
    //_logger.LogVector(expectedLayer->Values);
    double error = 0;
    for (int b = 0; b < finalLayer->GetRows(); b++) {
        error += 0.5 * (expectedLayer->GetValue(b) - finalLayer->GetValue(b)) * (expectedLayer->GetValue(b) - finalLayer->GetValue(b));
        _logger.LogMessage("Temp error is ");
        _logger.LogNumber(error);
        _logger.LogNewline();
    }

    _dError_dActivatedOutput = *finalLayer - *expectedLayer;
    _logger.LogLine("Calculated derivatives for final layer.");
    _logger.LogMessage("Error is: ");
    _logger.LogNumber(error);
    _logger.LogNewline();

    return error;
}

void NetworkTrainer::GetdError_dActivatedOutput(Matrix* weightMatrix, PyNet::Models::Vector* inputLayer, PyNet::Models::Vector* outputLayer) {
    //dError_dOutputCurrent.clear();
    _logger.LogLine("Calculating error derivative with respect to current output layer.");
    _logger.LogNumber(weightMatrix->GetCols());
    _logger.LogNewline();

    auto& dActivatedLayerAbove_dOutput = _context.get<Vector>();
    outputLayer->CalculateActivationDerivative(dActivatedLayerAbove_dOutput);

    auto& dError_dOutput = _context.get<Vector>();
    dError_dOutput = _dError_dActivatedLayerAbove ^ dActivatedLayerAbove_dOutput;

    auto& weightMatrixTranspose = _context.get<Matrix>();
    weightMatrixTranspose = ~*weightMatrix;
    //*dError_dActivatedInput = (*dError_dOutput_transpose) * *weightMatrix;
    _dError_dActivatedOutput = weightMatrixTranspose * dError_dOutput;

    _logger.LogMessage("dError_dOutputCurrent: ");
    //_logger.LogVector(dError_dOutputCurrent);
}

void NetworkTrainer::UpdateWeights(std::vector<Matrix*> weightMatrices, std::vector<Vector*> biases, double learningRate) {
    _logger.LogLine("Updating weights...");
    for (int weightMatrixIndex = static_cast<int>(weightMatrices.size() - 1); weightMatrixIndex >= 0; weightMatrixIndex--) {
        Matrix* weightMatrix = weightMatrices[weightMatrixIndex];
        Vector* bias = biases[weightMatrixIndex];

        *bias = *bias - (*_adjustmentCalculator.GetBiasAdjustment(weightMatrixIndex) * learningRate);
        *weightMatrix = *weightMatrix - (*_adjustmentCalculator.GetWeightAdjustment(weightMatrixIndex) * learningRate);
    }

    _adjustmentCalculator.SetNewBatch(true);
}

void NetworkTrainer::GetAdjustmentsForWeightMatrix(Matrix* weightMatrix, Vector* inputLayer, Vector* outputLayer, int weightMatrixIndex) {
    _logger.LogMessage("Calcuating loop for weight matrix: ");
    _logger.LogNumber(weightMatrixIndex);
    _logger.LogNewline();

    _logger.LogLine("Calculating adjustments.");

    auto& temp_name = _context.get<Vector>();
    outputLayer->CalculateActivationDerivative(temp_name);

    auto dError_dBias = 0.0;
    dError_dBias = _dError_dActivatedOutput | temp_name;
    _adjustmentCalculator.AddBiasAdjustment(weightMatrixIndex, dError_dBias);

    /// /////////////////////////////////

    auto& dActivatedOutput_dOutput = _context.get<Vector>();
    outputLayer->CalculateActivationDerivative(dActivatedOutput_dOutput);

    auto& dError_dOutput = _context.get<Matrix>();
    dError_dOutput = _dError_dActivatedOutput ^ dActivatedOutput_dOutput;

    auto dOutput_dWeight = inputLayer;

    auto& dOutput_dWeight_Transpose = _context.get<Matrix>();
    dOutput_dWeight_Transpose = ~*dOutput_dWeight;
    auto& dError_dWeight = _context.get<Matrix>();
    dError_dWeight = dError_dOutput * dOutput_dWeight_Transpose;

    _adjustmentCalculator.AddWeightAdjustment(weightMatrixIndex, dError_dWeight);

    _dError_dActivatedLayerAbove = _dError_dActivatedOutput;
    GetdError_dActivatedOutput(weightMatrix, inputLayer, outputLayer);
}

void NetworkTrainer::GetAdjustments(std::vector<Matrix*> weightMatrices, std::vector<Vector*> layers) {
    for (int a = static_cast<int>(weightMatrices.size() - 1); a >= 0; a--) {
        GetAdjustmentsForWeightMatrix(weightMatrices[a], layers[a], layers[(size_t)a+1], a);
    }

    _logger.LogLine("End of GetAdjustments");
}