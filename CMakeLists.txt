CMAKE_MINIMUM_REQUIRED(VERSION 3.27)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(UseNative True)

if(UseNative)
    message("Use native architecture")
    set(CMAKE_CUDA_ARCHITECTURES "native")
else()
    message("Use virtual architecture.")
    set(CMAKE_CUDA_ARCHITECTURES "86-virtual")
endif()

set(CMAKE_CXX_STANDARD 20)
project(narrow_kernel_mpm CUDA CXX)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -G --expt-relaxed-constexpr --extended-lambda")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 --expt-relaxed-constexpr --extended-lambda")


# include_directories(${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party/partio/src/lib)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party/spdlog/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party/json/include)

add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/partio)
add_subdirectory(src)

enable_testing()

add_subdirectory(tests)
