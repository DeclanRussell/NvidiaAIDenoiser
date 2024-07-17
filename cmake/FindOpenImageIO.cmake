FIND_PATH(OIIO_INCLUDE_DIR OpenImageIO/imageio.h
    $ENV{OIIO_DIR}/include
    $ENV{OIIO_DIR}
    ${OIIO_DIR}/include
    ${OIIO_DIR}
    ~/Library/Frameworks
    /Library/Frameworks
    /usr/local/include
    /usr/include
    /sw/include
    /opt/local/include
    /opt/csw/include
    /opt/include
    /usr/freeware/include
)

FIND_LIBRARY(OIIO_LIBRARY
             NAMES OpenImageIO
             PATHS 
             $ENV{OIIO_DIR}/lib
             ${OIIO_DIR}/lib
             /usr/lib 
             /usr/local/lib
             )

IF (OIIO_INCLUDE_DIR AND OIIO_LIBRARY)
    SET(OIIO_FOUND TRUE)
    SET(OIIO_LIBRARY_DIR ${OIIO_LIBRARY})
ELSE (OIIO_INCLUDE_DIR AND OIIO_LIBRARY)
    SET(OIIO_FOUND)
    SET(OIIO_LIBRARY_DIR)
ENDIF (OIIO_INCLUDE_DIR AND OIIO_LIBRARY)
