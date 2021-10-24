cmake_minimum_required( VERSION 3.20 )

project(PyNet.Infrastructure)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(C_STANDARD C17)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

#include(FetchContent)
#FetchContent_Declare(
 # googletest
#  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
#)
# For Windows: Prevent overriding the parent project's compiler/linker settings
#set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#FetchContent_MakeAvailable(googletest)

#enable_testing()
#include(googletest)


# include files
include_directories( ./include ./libs/PyNet.Models/include ./tests )

# target
add_library(PyNet.Infrastructure SHARED
	./include/PyNetwork_Functions.h
	./src/Settings.h
	./src/Context.h
	./src/AdjustmentCalculator.h
	./src/AdjustmentCalculator.cpp 
	./src/LayerPropagator.h
	./src/LayerPropagator.cpp 
	./src/Logger.h
	./src/Logger.cpp 
	./src/NetworkTrainer.h
	./src/NetworkTrainer.cpp
	./src/PyNetwork.h 
	./src/PyNetwork.cpp 
	./src/PyNetwork_Functions.cpp
	./tests/LayerPropagatorTests.cpp)

add_executable(PyNet.Console ./PyNet.Console/PyNet.Console.cpp)

target_link_libraries(PyNet.Console PyNet.Infrastructure )

# 3rd party libs
add_subdirectory( ./libs/PyNet.Models )
target_link_libraries( PyNet.Infrastructure PyNet.Models )
#target_link_libraries( PyNet.Infrastructure gtest_main )

#gtest_discover_tests( PyNet.Infrastructure )