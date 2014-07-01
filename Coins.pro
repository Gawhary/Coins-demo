#-------------------------------------------------
#
# Project created by QtCreator 2014-06-25T02:25:16
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = Coins
CONFIG   += console
CONFIG   -= app_bundle
CONFIG  += C++11

TEMPLATE = app


SOURCES += \
    Coins.cpp

INCLUDEPATH += "./opencv/include" \
                "./tesseract/include"

LIBS += -LE:\OpenCV\OpenCV2.4.6\build\x86\mingw\lib \
    -lopencv_core246 \
    -lopencv_highgui246 \
    -lopencv_imgproc246 \
    -lopencv_features2d246 \
    -lopencv_calib3d246 \
