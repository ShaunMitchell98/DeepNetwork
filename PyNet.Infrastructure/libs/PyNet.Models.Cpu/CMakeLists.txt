project(PyNet.Models.Cpu)

# include files
include_directories( ./include/PyNet.Models.Cpu ./src )
include_directories(${CMAKE_SOURCE_DIR}/libs/PyNet.Models)

# target
add_library(PyNet.Models.Cpu STATIC 
	./include/PyNet.Models.Cpu/CpuMatrix.h
	./src/CpuMatrix.cpp
	./include/PyNet.Models.Cpu/CpuModule.h)

target_link_libraries(PyNet.Models.Cpu PyNet.Models)