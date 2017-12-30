TARGET=Denoiser
OBJECTS_DIR=obj

win32:DEFINES+=WIN32
win32:DEFINES+= NOMINMAX

CONFIG-=app_bundle
CONFIG += c++11
SOURCES += \
    src/main.cpp

INCLUDEPATH +=./contrib/optix/include
INCLUDEPATH +=$$(CUDA_PATH)/include
INCLUDEPATH +=./contrib/DevIL/include
LIBS += -L./contrib/optix/bin
LIBS += -loptix.1
LIBS += -L./contrib/DevIL/lib
LIBS += -lDevIL -lILU -lILUT
DESTDIR=./bin

DEPENDPATH+=include





