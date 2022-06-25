include(FetchContent)

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)
FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

add_executable(PyNet.Models.Cpu.Tests CpuMatrixTests.cpp)

add_dependencies(PyNet.Models.Cpu.Tests gtest)
include_directories(${source_dir}/include)
target_link_libraries(PyNet.Models.Cpu.Tests
 PRIVATE
  GTest::GTest
  PyNet.Models.Cpu)

add_test(InitialTest PyNet.Models.Cpu.Tests)