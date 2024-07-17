# Looks for the environment variable:
# OPTIX80_PATH

# Sets the variables :
# OPTIX80_INCLUDE_DIR

# OptiX80_FOUND

set(OPTIX80_PATH $ENV{OPTIX80_PATH})

if("${OPTIX80_PATH}" STREQUAL "")
    if(WIN32)
        # Try finding it inside the default installation directory under Windows first.
        set(OPTIX80_PATH "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0")
    else()
        # Adjust this if the OptiX SDK 8.0.0 installation is in a different location.
        set(OPTIX80_PATH "~/NVIDIA-OptiX-SDK-8.0.0-linux64")
    endif()
endif()

find_path(OPTIX80_INCLUDE_DIR optix_host.h ${OPTIX80_PATH}/include)

# message("OPTIX80_INCLUDE_DIR = " "${OPTIX80_INCLUDE_DIR}")
include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(OptiX80 DEFAULT_MSG OPTIX80_INCLUDE_DIR)

mark_as_advanced(OPTIX80_INCLUDE_DIR)

# message("OptiX80_FOUND = " "${OptiX80_FOUND}")