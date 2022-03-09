project(PyNet.Console)

# include files
include_directories( ./modules)

add_executable(PyNet.Console ./modules/Console.ixx)
target_link_libraries( PyNet.Console PyNet.Infrastructure )