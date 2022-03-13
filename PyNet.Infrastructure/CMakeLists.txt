cmake_minimum_required( VERSION 3.20 )

project(PyNet.Infrastructure)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_SYSROOT_FLAG_CODE "list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS ixx)")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(C_STANDARD C17)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# include files
include_directories(
	 ./tests
	 ./modules
	"C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/VS/UnitTest/include"
)

# target
add_library(PyNet.Infrastructure SHARED
	./modules/PyNetwork_Functions.ixx
	./modules/Settings.ixx
	./modules/AdjustmentCalculator.ixx
	./modules/InfrastructureModule.ixx
	./modules/Logger.ixx
	./modules/SteepestDescent.ixx
	./modules/PyNetwork.ixx
	./modules/QuadraticLoss.ixx
	./modules/LayerNormaliser.ixx
	./modules/Setup.ixx
	./modules/XmlWriter.ixx
	./modules/XmlReader.ixx
	./modules/Xml_Functions.ixx
	./modules/NetworkRunner.ixx
	./modules/NetworkTrainer.ixx
	./modules/VariableLearningSettings.ixx
	./modules/TrainingAlgorithm.ixx
	./modules/GradientCalculator.ixx
	./modules/LayerPropagator.ixx
	./modules/PyNet.Infrastructure.ixx
)

# 3rd party libs
add_subdirectory( ./libs/PyNet.Models )
add_subdirectory( ./libs/PyNet.Models.Cuda )
add_subdirectory( ./libs/PyNet.Models.Cpu )
add_subdirectory( ./libs/PyNet.DI)
add_subdirectory( ./libs/PyNet.Console )
target_link_libraries( PyNet.Infrastructure PyNet.Models )
target_link_libraries( PyNet.Infrastructure PyNet.Models.Cuda )
target_link_libraries( PyNet.Infrastructure PyNet.Models.Cpu )
target_link_libraries( PyNet.Infrastructure PyNet.DI )
target_link_libraries(PyNet.Console PyNet.Infrastructure )

add_library(PyNet.Infrastructure.Tests SHARED
		./tests/LayerPropagatorTests.ixx
	./tests/SteepestDescentTests.ixx
	./tests/PyNetworkTests.ixx
	./tests/UnitTest.ixx
)

target_link_libraries(PyNet.Infrastructure.Tests PyNet.Infrastructure)
link_directories("C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/VS/UnitTest/lib/x64/Microsoft.VisualStudio.TestTools.CppUnitTestFramework.lib")
