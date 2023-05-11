project(PyNet.DI)

# include files
include_directories( ./include/PyNet.DI )
include_directories( ./src/Internal )

# target
add_library(PyNet.DI STATIC 
    ./include/PyNet.DI/Context.h
    ./include/PyNet.DI/ContextBuilder.h
	./include/PyNet.DI/InstanceMode.h
	./include/PyNet.DI/ServiceDescriptor.h
	./include/PyNet.DI/IServiceDescriptor.h
	./include/PyNet.DI/IServiceCollection.h
	./include/PyNet.DI/ServiceCollection.h
	./include/PyNet.DI/ServiceProvider.h
	./include/PyNet.DI/IServiceProvider.h
	./include/PyNet.DI/ItemRegistrar.h
	./include/PyNet.DI/Module.h
	./src/ContextBuilder.cpp
	./src/Internal/ServiceFactory.h)