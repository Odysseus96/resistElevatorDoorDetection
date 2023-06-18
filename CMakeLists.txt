cmake_minimum_required(VERSION 3.0)
project(yolov8)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "-std=c++11 \
-fomit-frame-pointer -fstrict-aliasing -ffunction-sections \
-fdata-sections -ffast-math -fpermissive")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_C_COMPILER /usr/bin/clang)
set(CMAKE_CXX_COMPILER /usr/bin/clang++)

find_package(OpenCV REQUIRED)

find_package(OpenVINO REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    /opt/homebrew/Cellar/openvino/2023.0.0/include
)

set(OpenVINO_LIB_PATH /opt/homebrew/Cellar/openvino/2023.0.0/lib)

# link_directories(${OpenVINO_LIB_PATH})

message(STATUS "OpenCV library status:")
message(STATUS "version:${OpenCV_VERSION}")
message(STATUS "libraries:${OpenCV_LIBS}")
message(STATUS "include path:${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenVINO: ${OpenVINO_INCLUDE_DIRS}")

# link_directories(${OpenCV_LIBS})
aux_source_directory(${CMAKE_SOURCE_DIR}/src SRC)

add_executable(yolov8 ${SRC})

target_link_directories(yolov8
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(yolov8
    ${OpenCV_LIBS}
)