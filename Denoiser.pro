TARGET=Denoiser
OBJECTS_DIR=obj
# as I want to support 4.8 and 5 this will set a flag for some of the mac stuff
# mainly in the types.h file for the setMacVisual which is native in Qt5
isEqual(QT_MAJOR_VERSION, 5) {
        cache()
        DEFINES +=QT5BUILD
}
UI_HEADERS_DIR=ui
MOC_DIR=moc
win32:DEFINES+=WIN32
win32:DEFINES+= NOMINMAX

CONFIG-=app_bundle
CONFIG += c++11
QT+= core
SOURCES += \
    src/main.cpp

INCLUDEPATH +=./contrib/optix/include
INCLUDEPATH +=$$(CUDA_PATH)/include
LIBS += -L./contrib/optix/bin
LIBS += -loptix.1
DESTDIR=./bin

CONFIG += console
CONFIG -= app_bundle

DEPENDPATH+=include





