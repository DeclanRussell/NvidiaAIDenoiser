cmake_minimum_required(VERSION 3.10)
project(Denoiser)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

find_package(OptiX80)
# Add CUDA
find_package(CUDAToolkit 12.0 REQUIRED)
find_package(OpenImageIO REQUIRED)

# Include directories
include_directories(${OPTIX80_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})

# Source files
set(SOURCES src/main.cpp)

# Executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} ${OIIO_LIBRARY} CUDA::cudart)
