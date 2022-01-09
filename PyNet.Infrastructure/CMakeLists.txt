cmake_minimum_required( VERSION 3.20 )

project(PyNet.Infrastructure)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(C_STANDARD C17)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# include files
include_directories( ./include 
	./libs/PyNet.Models/include
	./libs/PyNet.Models.Cuda/include
	./libs/PyNet.Models.Cpu/include
	./libs/PyNet.DI/include
	 ./tests
	 ./src
	./libs/di/src
	"C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/VS/UnitTest/include"
)

# target
add_library(PyNet.Infrastructure SHARED
	./include/PyNetwork_Functions.h
	./src/Settings.h
	./src/AdjustmentCalculator.h
	./src/AdjustmentCalculator.cpp 
	./src/InfrastructureModule.h
	./src/LayerPropagator.h
	./src/LayerPropagator.cpp 
	./src/Logger.h
	./src/Logger.cpp 
	./src/SteepestDescent.h
	./src/SteepestDescent.cpp
	./src/PyNetwork.h 
	./src/PyNetwork.cpp 
	./src/PyNetwork_Functions.cpp
	./src/QuadraticLoss.h
	./src/QuadraticLoss.cpp
	./src/Setup.h
	./src/XmlWriter.h
	./src/XmlWriter.cpp
	./src/XmlReader.h
	./src/XmlReader.cpp
	./src/VariableLearningSettings.h
	./src/TrainingAlgorithm.h
	./src/GradientCalculator.h
	./src/GradientCalculator.cpp
)

add_executable(PyNet.Console ./PyNet.Console/PyNet.Console.cpp)

target_link_libraries(PyNet.Console PyNet.Infrastructure )

# 3rd party libs
add_subdirectory( ./libs/PyNet.Models )
add_subdirectory( ./libs/PyNet.Models.Cuda )
add_subdirectory( ./libs/PyNet.Models.Cpu )
add_subdirectory( ./libs/PyNet.DI)
target_link_libraries( PyNet.Infrastructure PyNet.Models )
target_link_libraries( PyNet.Infrastructure PyNet.Models.Cuda )
target_link_libraries( PyNet.Infrastructure PyNet.Models.Cpu )
target_link_libraries( PyNet.Infrastructure PyNet.DI )


add_library(PyNet.Infrastructure.Tests SHARED
		./tests/LayerPropagatorTests.cpp
	./tests/NetworkTrainerTests.cpp
	./tests/PyNetworkTests.cpp
	./tests/UnitTest.h
)

include_directories(
	"C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/VS/UnitTest/include"
)

target_link_libraries(PyNet.Infrastructure.Tests PyNet.Infrastructure)
link_directories("C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/VS/UnitTest/lib/x64/Microsoft.VisualStudio.TestTools.CppUnitTestFramework.lib")
