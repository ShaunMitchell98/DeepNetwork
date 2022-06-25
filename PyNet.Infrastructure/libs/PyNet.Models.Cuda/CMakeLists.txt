project(PyNet.Models.Cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# include files
include_directories( ./include/PyNet.Models.Cuda ./src)

# target
add_library(PyNet.Models.Cuda SHARED
	./include/PyNet.Models.Cuda/Matrix_Operations.h
	./src/Matrix_Multiplication.cu
	./include/PyNet.Models.Cuda/CudaMatrix.h
	./src/CudaMatrix.cpp
	./src/CudaArray.h
	./src/Matrix_Subtraction.cu
	./src/Matrix_Addition_Assignment.cu
	./src/Matrix_Double_Multiplication.cu
	./src/Matrix_Double_Addition.cu
	./src/Matrix_Reciprocal.cu
	./src/Matrix_Addition.cu
	./src/Matrix_Max.cu
	./src/Matrix_Step.cu
	./src/Matrix_Hadamard.cu
	./src/Matrix_Exp.cu
	./src/Matrix_Dot_Product.cu
	./include/PyNet.Models.Cuda/CudaModule.h)

target_link_libraries(PyNet.Models.Cuda PyNet.Models)
target_link_libraries(PyNet.Models.Cuda PyNet.DI)
set_target_properties(PyNet.Models.Cuda PROPERTIES CUDA_ARCHITECTURES "75")
set_property(TARGET PyNet.Models.Cuda PROPERTY CUDA_STANDARD 17)

#add_library(PyNet.Models.Cuda.Tests SHARED
#	./tests/CudaMatrixTests.ixx
#	./tests/UnitTest.ixx
#)

#target_link_libraries(PyNet.Models.Cuda.Tests PyNet.Models.Cuda)