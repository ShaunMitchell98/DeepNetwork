project(PyNet.Models)

# include files
include_directories( ./include/PyNet.Models ./src )

# target
add_library(PyNet.Models STATIC 
	./include/PyNet.Models/ILogger.h
	./include/PyNet.Models/Loss.h
	./include/PyNet.Models/Matrix.h
	./src/Matrix.cpp)