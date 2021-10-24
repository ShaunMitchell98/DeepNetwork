project(PyNet.Infrastructure.Cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# include files
include_directories( ./include/PyNet.Infrastructure.Cuda ./src)

# target
add_library(PyNet.Infrastructure.Cuda SHARED 
	./include/PyNet.Infrastructure.Cuda/Matrix_Multiplication.h
	./src/Matrix_Multiplication.cu)

set_target_properties(PyNet.Infrastructure.Cuda PROPERTIES CUDA_ARCHITECTURES "75")
set_property(TARGET PyNet.Infrastructure.Cuda PROPERTY CUDA_STANDARD 17)