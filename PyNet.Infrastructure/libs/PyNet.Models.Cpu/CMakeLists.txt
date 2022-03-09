project(PyNet.Models.Cpu)

# target
add_library(PyNet.Models.Cpu STATIC 
	./modules/CpuMatrix.ixx
	./modules/CpuVector.ixx
	./modules/CpuLogistic.ixx
	./modules/CpuModule.ixx
	./modules/PyNet.Models.Cpu.ixx)

target_link_libraries(PyNet.Models.Cpu PyNet.Models)
target_link_libraries(PyNet.Models.Cpu PyNet.DI)