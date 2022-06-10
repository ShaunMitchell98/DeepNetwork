cmake_minimum_required( VERSION 3.20 )

find_package(CUDAToolkit)

IF (CUDAToolkit_FOUND)
  project(PyNet.Infrastructure LANGUAGES CXX CUDA)
ELSE()
  project(PyNet.Infrastructure LANGUAGES C CXX)
ENDIF()

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(C_STANDARD C17)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set (CMAKE_POLICY_DEFAULT_CMP0025 NEW)

# include files
include_directories( ./include 
	./libs/PyNet.Models/include
	./libs/PyNet.Models.Cpu/include
	./libs/PyNet.DI/include
	./build/gtest/src/gtest/googletest/include
	 ./tests
	 ./src
	./libs/di/src

	IF (CUDAToolkit_FOUND)
  		./libs/PyNet.Models.Cuda/include
	ENDIF()
)

# target
add_library(PyNet.Infrastructure SHARED
	./include/PyNetwork_Functions.h
	./src/Headers.h
	./src/Settings.h
	./src/AdjustmentCalculator.h
	./src/AdjustmentCalculator.cpp 
	./src/InfrastructureModule.h
	./src/Intermediary.h
	./src/Logger.h
	./src/Logger.cpp 
	./src/SteepestDescent.h
	./src/SteepestDescent.cpp
	./src/PyNetwork.h 
	./src/PyNetwork_Functions.cpp
	./src/QuadraticLoss.h
	./src/QuadraticLoss.cpp
	./src/Startup.h
	./src/Startup.cpp
	./src/XmlWriter.h
	./src/XmlWriter.cpp
	./src/XmlReader.h
	./src/XmlReader.cpp
	./src/TrainingAlgorithm.h
	./src/BackPropagator.h
	./src/BackPropagator.cpp
	./src/NetworkRunner.h
	./src/NetworkRunner.cpp
	./src/NetworkTrainer.h
	./src/NetworkTrainer.cpp
	./src/Xml_Functions.h
	./src/Xml_Functions.cpp
	./src/BernoulliGenerator.h
	./src/BernoulliGenerator.cpp
	./src/Layers/Layer.h
	./src/Layers/DenseLayer.h
	./src/Layers/ConvolutionalLayer.h
	./src/Layers/DropoutLayer.h
	./src/Layers/LayerModule.h
	./src/Layers/MaxPoolingLayer.h
	./src/Layers/InputLayer.h
	./src/Layers/FlattenLayer.h
	./src/Layers/TrainableLayer.h
	./src/ReceptiveFieldProvider.h
	./src/MatrixPadder.h
	./src/Activations/Activation.h
	./src/Activations/Logistic.h
	./src/Activations/Relu.h
	./src/Activations/ActivationModule.h
	./src/Layers/SoftmaxLayer.h
	./src/VLService.h
	./src/VLService.cpp
	./src/TrainingState.h
)

# 3rd party libs

IF (CUDAToolkit_FOUND)
	add_subdirectory( ./libs/PyNet.Models.Cuda )
ENDIF()

add_subdirectory( ./libs/PyNet.Models )
add_subdirectory( ./libs/PyNet.Models.Cpu )
add_subdirectory( ./libs/PyNet.DI)
add_subdirectory( ./libs/PyNet.Console )
add_subdirectory( ./libs/PyNet.Models.Cpu/tests )

add_subdirectory( ./tests)

IF (CUDAToolkit_FOUND)
	add_subdirectory( ./libs/PyNet.Models.Cuda/tests )
	target_link_libraries( PyNet.Infrastructure PyNet.Models.Cuda )
ENDIF()

target_link_libraries( PyNet.Infrastructure PyNet.Models )
target_link_libraries( PyNet.Infrastructure PyNet.Models.Cpu )
target_link_libraries( PyNet.Infrastructure PyNet.DI )
target_link_libraries(PyNet.Console PyNet.Infrastructure )

enable_testing()
