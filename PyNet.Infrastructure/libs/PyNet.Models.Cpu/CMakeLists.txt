project(PyNet.Models.Cpu)

# include files
include_directories( ./include/PyNet.Models.Cpu ./src )

# target
add_library(PyNet.Models.Cpu STATIC 
	./include/PyNet.Models.Cpu/CpuMatrix.h
	./src/CpuMatrix.cpp
	./include/PyNet.Models.Cpu/CpuLogistic.h
	./src/CpuLogistic.cpp
	./include/PyNet.Models.Cpu/ActivationProvider.h)