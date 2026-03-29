cmake_minimum_required(VERSION 3.10)
project(Denoiser)

set(CMAKE_CXX_STANDARD 20)
include(GNUInstallDirs)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# Add CUDA
find_package(CUDA 12.9 REQUIRED)

find_package(OpenImageIO CONFIG REQUIRED)

message(STATUS "Found OpenImageIO: ${OpenImageIO_DIR}"
               " (includes: ${OpenImageIO_INCLUDE_DIRS}, library: ${OpenImageIO_LIBRARIES})")
message(STATUS "Found CUDA: ${CUDA_TOOLKIT_ROOT_DIR}"
               " (includes: ${CUDA_INCLUDE_DIRS}, library: ${CUDA_cudart_static_LIBRARY})")

# Include directories
include_directories("contrib/optix/include"
                    ${CUDA_INCLUDE_DIRS})

# Source files
set(SOURCES src/main.cpp)

# Executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
                      ${CUDA_cudart_static_LIBRARY}
                      openimageio::openimageio)

# Install app and Conan OpenImageIO artifacts explicitly.
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

