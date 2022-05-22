
include(ExternalProject)
ExternalProject_Add(gtest
  URL https://github.com/google/googletest/archive/release-1.11.0.zip
  # Comment above line, and uncomment line below to use subversion.
  # SVN_REPOSITORY http://googletest.googlecode.com/svn/trunk/ 
  # Uncomment line below to freeze a revision (here the one for 1.7.0)
  # SVN_REVISION -r700
  
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
  INSTALL_COMMAND ""
)
ExternalProject_Get_Property(gtest source_dir binary_dir)

add_library(PyNet.Models.Cpu.Tests SHARED CpuMatrixTests.cpp)

add_dependencies(PyNet.Models.Cpu.Tests gtest)
include_directories(${source_dir}/include)
#gtest_discover_tests(PyNet.Models.Cpu.Tests)
#target_link_libraries(PyNet.Models.Cpu.Tests ./build/gtest/src/gtest-build/lib/gtest)
#target_link_libraries(PyNet.Models.Cpu.Tests ./build/gtest/src/gtest-build/lib/gtest_main)
target_link_libraries(PyNet.Models.Cpu.Tests ./gtest/src/gtest-build/lib/Release/gtest)
target_link_libraries(PyNet.Models.Cpu.Tests ./gtest/src/gtest-build/lib/Release/gtest_main)
enable_testing()
add_test(NAME Test1
COMMAND PyNet.Models.Cpu.Tests)