cmake_minimum_required(VERSION 3.8)
project(perception)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(crp_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(autoware_perception_msgs REQUIRED)

# ---------------------------------------------------------------------------
# ONNX Runtime (C++). Telepítés: a hivatalos release tar.gz kicsomagolása, pl.
#   onnxruntime-linux-x64-gpu-<ver>.tgz  (CUDA-val)  vagy  onnxruntime-linux-x64-<ver>.tgz (CPU)
# majd:  colcon build --cmake-args -DONNXRUNTIME_ROOT=/opt/onnxruntime
# ---------------------------------------------------------------------------
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime telepítési könyvtár (include/ és lib/)")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT}/include
  PATH_SUFFIXES onnxruntime onnxruntime/core/session)
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS ${ONNXRUNTIME_ROOT}/lib)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "ONNX Runtime nem található. Add meg: -DONNXRUNTIME_ROOT=<útvonal>")
endif()
get_filename_component(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_LIBRARY} DIRECTORY)
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")

include_directories(include)

# ROS-független rész: inferencia + döntés (külön is tesztelhető)
add_library(perception_core SHARED
  src/centerpoint_detector.cpp
  src/brake_decision.cpp
)
target_include_directories(perception_core PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(perception_core ${ONNXRUNTIME_LIBRARY})
set_target_properties(perception_core PROPERTIES
  INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
  BUILD_WITH_INSTALL_RPATH TRUE)

add_executable(perception_node src/perceptionNode.cpp)
target_link_libraries(perception_node perception_core)
ament_target_dependencies(perception_node
  rclcpp
  sensor_msgs
  crp_msgs
  std_msgs
  autoware_perception_msgs
)
set_target_properties(perception_node PROPERTIES
  INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR};${CMAKE_INSTALL_PREFIX}/lib")

# Offline ellenőrző eszköz (nuScenes .bin -> JSON detekciók)
add_executable(centerpoint_cli tools/centerpoint_cli.cpp)
target_link_libraries(centerpoint_cli perception_core)
set_target_properties(centerpoint_cli PROPERTIES
  INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR};${CMAKE_INSTALL_PREFIX}/lib")

install(TARGETS perception_core
  LIBRARY DESTINATION lib
)

install(TARGETS
  perception_node
  centerpoint_cli
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  launch
  models
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/export_centerpoint_onnx.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
