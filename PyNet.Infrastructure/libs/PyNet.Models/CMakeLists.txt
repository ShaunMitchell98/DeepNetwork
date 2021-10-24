project(PyNet.Models)

# include files
include_directories( ./include/PyNet.Models ./src ./libs/PyNet.Infrastructure.Cuda/include)

# target
add_library(PyNet.Models STATIC 
	./include/PyNet.Models/ILogger.h
	./include/PyNet.Models/Activation.h
	./src/Activation.cpp
	./include/PyNet.Models/Logistic.h
	./src/Logistic.cpp
	./include/PyNet.Models/Matrix.h
	./src/Matrix.cpp
	./include/PyNet.Models/Vector.h
	./src/Vector.cpp)

add_subdirectory( ./libs/PyNet.Infrastructure.Cuda)
target_link_libraries( PyNet.Models PyNet.Infrastructure.Cuda )