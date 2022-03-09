project(PyNet.Models)

# target
add_library(PyNet.Models STATIC 
	./modules/ILogger.ixx
	./modules/Loss.ixx
	./modules/Activation.ixx
	./modules/Matrix.ixx
	./modules/Vector.ixx
	./modules/PyNet.Models.ixx)