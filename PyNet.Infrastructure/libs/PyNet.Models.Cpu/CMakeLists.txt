project(PyNet.Models.Cpu)

# include files
include_directories( ./include/PyNet.Models.Cpu ./src ./tests)
include_directories(${CMAKE_SOURCE_DIR}/libs/PyNet.Models)
#include_directories(${GTEST_INCLUDE_DIRS})
#include_directories(C:\Users\Shaun Mitchell\source\repos\PyNet\PyNet.Infrastructure\build\gtest\src\gtest\googletest\include\gtest)
# set(GTEST_ROOT "" CACHE PATH "Path to googletest")
#include(GoogleTest)

# target
add_library(PyNet.Models.Cpu STATIC 
	./include/PyNet.Models.Cpu/CpuMatrix.h
	./src/CpuMatrix.cpp
	./include/PyNet.Models.Cpu/CpuModule.h)


target_link_libraries(PyNet.Models.Cpu PyNet.Models)


#add_dependencies(PyNet.Models.Cpu.Tests gtest)
#include_directories(${source_dir}/include)
#target_link_libraries(PyNet.Models.Cpu.Tests ${binary_dir}/libgtest.a)
#target_link_libraries(PyNet.Models.Cpu.Tests ${binary_dir}/libgtest_main.a)

#enable_testing()

#find_package(GoogleTest)
#gtest_discover_tests(PyNet.Models.Cpu.Tests)