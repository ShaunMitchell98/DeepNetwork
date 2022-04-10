project(PyNet.DI)

# include files
include_directories( ./include/PyNet.DI )

# target
add_library(PyNet.DI STATIC 
    ./include/PyNet.DI/Context.h
    ./include/PyNet.DI/ContextBuilder.h
	./include/PyNet.DI/InstanceMode.h
	./include/PyNet.DI/Item.h
	./include/PyNet.DI/ItemContainer.h     
	./include/PyNet.DI/ItemRegistrar.h
	./include/PyNet.DI/Module.h
	./src/ContextBuilder.cpp)