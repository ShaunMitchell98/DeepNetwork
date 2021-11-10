project(PyNet.Models.Cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# include files
include_directories( ./include/PyNet.Models.Cuda ./src)
include_directories(${CMAKE_SOURCE_DIR}/libs/PyNet.Models)

# target
add_library(PyNet.Models.Cuda SHARED 
	./include/PyNet.Models.Cuda/Matrix_Multiplication.h
	./src/Matrix_Multiplication.cu
	./include/PyNet.Models.Cuda/CudaMatrix.h
	./src/CudaMatrix.cpp)

target_link_libraries(PyNet.Models.Cuda PyNet.Models)
set_target_properties(PyNet.Models.Cuda PROPERTIES CUDA_ARCHITECTURES "75")
set_property(TARGET PyNet.Models.Cuda PROPERTY CUDA_STANDARD 17)