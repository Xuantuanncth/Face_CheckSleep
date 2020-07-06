cmake_minimum_required(VERSION 3.5)
project(main)

add_definitions(-std=c++11)
#set(QMAKE_CXXFLAGS "-std=c++11")
#include_directories(/opt/OpenBLAS/include)
include_directories(/usr/local/dlib/)
#include_directories(/home/pi/OpenBLAS)
#link_directories(/usr/local/cuda/lib64)
link_directories(/usr/local/dlib/lib)
#include_directories(/home/pi/OpenBLAS)

find_package( OpenCV REQUIRED)
find_package(Threads REQUIRED) 
find_library(WIRINGPI_LIBRARIES NAMES wiringPi)

AUX_SOURCE_DIRECTORY(./src DIR_SRCS)
add_executable( main ${DIR_SRCS}  src/demo.cpp)

set(CMAKE_CXX_FLAGS "-lwiringPi")
set(CMAKE_CXX_FLAGS  "-lpthread")
set(CMAKE_CXX_FLAGS  "-O3")
target_link_libraries( main ${OpenCV_LIBS} )
target_link_libraries( main openblas)
#target_link_libraries( main cudart)
#target_link_libraries( main nvinfer)
#target_link_libraries( main nvparsers)
target_link_libraries( main dlib)
target_link_libraries(main ${WIRINGPI_LIBRARIES})


