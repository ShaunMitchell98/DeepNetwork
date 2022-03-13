project(PyNet.DI)

# include files
include_directories( ./include/PyNet.DI )

# target
add_library(PyNet.DI STATIC 
	./modules/Module.ixx
	./modules/Context.ixx
	./modules/Item.ixx
	./modules/InstanceMode.ixx
	./modules/ItemRegistrar.ixx
	./modules/ContextBuilder.ixx
	./modules/ItemContainer.ixx
	./modules/PyNet.DI.ixx)