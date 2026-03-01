cmake_minimum_required(VERSION 3.10)
project(Denoiser)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# Add CUDA
find_package(CUDA 12.0 REQUIRED)


add_subdirectory(contrib/OpenImageIO)

# Include directories
include_directories("contrib/optix/include"
                    "contrib/OpenImageIO/include"
                    ${CUDAToolkit_INCLUDE_DIRS})

# Source files
set(SOURCES src/main.cpp)

# Executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} CUDA_cudart_static_LIBRARY)
