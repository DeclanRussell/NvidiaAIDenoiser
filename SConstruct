
import os
PLATFORM = Platform()
DEBUG = ARGUMENTS.get("debug", 0)

CUDA_INCLUDE_PATH = os.environ['CUDA_PATH'] + "/include"

ENV = Environment(CPPPATH = ['.', "./contrib/optix/include", "./contrib/DevIL/include", CUDA_INCLUDE_PATH
],
                  CCFLAGS="-std=c++11")

# Used for debbugging
if int(DEBUG):
    ENV.Append(CCFLAGS=' -ggdb3')

SOURCES = Glob("src/*.cpp")

LIBPATH = []
if PLATFORM.name == "darwin":
    LIBPATH.append("/usr/local/lib")
if PLATFORM.name == "win32":
    LIBPATH.append("./contrib/optix/bin")
    LIBPATH.append("./contrib/DevIL/lib")
    

LIBS = []
LIBS.append("optix.1")
LIBS.append("DevIL")
LIBS.append("ILU")
LIBS.append("ILUT")

LINKFLAGS = []

if PLATFORM.name == "win32":
    ENV.Append(CPPDEFINES = "NOMINMAX")

# Copy over the dlls to the bin directory
ENV.Command("bin/DevIL.dll", "./contrib/DevIL/bin/DevIL.dll", Copy("$TARGET", "$SOURCE"))
ENV.Command("bin/ILU.dll", "./contrib/DevIL/bin/ILU.dll", Copy("$TARGET", "$SOURCE"))
ENV.Command("bin/ILUT.dll", "./contrib/DevIL/bin/ILUT.dll", Copy("$TARGET", "$SOURCE"))
ENV.Command("bin/cudart64_90.dll", "./contrib/optix/bin/cudart64_90.dll", Copy("$TARGET", "$SOURCE"))
ENV.Command("bin/cudnn64_7.dll", "./contrib/optix/bin/cudnn64_7.dll", Copy("$TARGET", "$SOURCE"))
ENV.Command("bin/optix.1.dll", "./contrib/optix/bin/optix.1.dll", Copy("$TARGET", "$SOURCE"))
ENV.Command("bin/optix_denoiser.dll", "./contrib/optix/bin/optix_denoiser.dll", Copy("$TARGET", "$SOURCE"))

program = ENV.Program(target="bin/Denoiser", source=SOURCES,
                              LIBPATH=LIBPATH, LIBS=LIBS, LINKFLAGS=LINKFLAGS)

