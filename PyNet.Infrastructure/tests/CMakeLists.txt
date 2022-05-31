include(FetchContent)

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        e649993a402d96afe25fbf3413749adf0f2947f6
)
FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

add_executable(PyNet.Infrastructure.Tests 
Layers/DenseLayerTests.cpp
Layers/InputLayerTests.cpp
Layers/FakeLayer.h
ContainerFixture.h
NetworkRunnerTests.cpp
BackPropagatorTests.cpp
IntegrationTestsModule.h
UnitTest.h)

add_dependencies(PyNet.Infrastructure.Tests gtest)
include_directories(${source_dir}/include)
target_link_libraries(PyNet.Infrastructure.Tests
 PRIVATE
  GTest::GTest
  PyNet.Infrastructure)

add_test(InitialTest PyNet.Infrastructure.Tests)