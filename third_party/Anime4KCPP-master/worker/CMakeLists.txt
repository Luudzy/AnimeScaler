cmake_minimum_required(VERSION 3.20)
project(anime4kcpp_worker LANGUAGES CXX)

set(ANIME4KCPP_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." CACHE PATH "Anime4KCPP source root")
set(ANIME4KCPP_BUILD "${ANIME4KCPP_ROOT}/build-opencl-md" CACHE PATH "Anime4KCPP build directory")

add_executable(anime4kcpp_worker main.cpp)
target_compile_features(anime4kcpp_worker PRIVATE cxx_std_17)
target_include_directories(anime4kcpp_worker PRIVATE
    "${ANIME4KCPP_ROOT}/core/include"
    "${ANIME4KCPP_BUILD}/core/include")
target_link_libraries(anime4kcpp_worker PRIVATE
    "${ANIME4KCPP_BUILD}/core/Release/ac.lib"
    "${ANIME4KCPP_BUILD}/_deps/opencl-build/external/OpenCL-ICD-Loader/Release/OpenCL.lib")
set_property(TARGET anime4kcpp_worker PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
set_target_properties(anime4kcpp_worker PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
