project(PyNet.Console)

# include files
include_directories( ./src )

add_executable(PyNet.Console ./src/PyNet.Console.cpp)
target_link_libraries( PyNet.Console PyNet.Infrastructure )