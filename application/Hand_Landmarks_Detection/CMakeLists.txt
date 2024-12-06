cmake_minimum_required(VERSION 3.24)

project(rpi-demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
string(REPLACE "-g" "-g3" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
string(REPLACE "-g" "-g3" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -g3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g3")

add_subdirectory(third_party/tensorflow/tensorflow/lite "${CMAKE_BINARY_DIR}/tflite")

find_package(OpenCV REQUIRED COMPONENTS core videoio imgproc highgui)
find_package(Optimium-Runtime REQUIRED HINTS "/workspace/optimium-runtime")

add_executable(rpi-demo main.cpp Recorder.cpp ModelRunner.cpp Postprocess.cpp TFLite.cpp Optimium.cpp nms.cpp)

target_link_libraries(rpi-demo PRIVATE
                      opencv_core 
                      opencv_videoio 
                      opencv_imgproc 
                      opencv_highgui
                      tensorflow-lite 
                      Optimium::Runtime)

