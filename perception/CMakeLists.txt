cmake_minimum_required(VERSION 3.8)
project(perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(crp_msgs REQUIRED)
find_package(std_msgs REQUIRED)

include_directories(include)

add_executable(perception_node src/perceptionNode.cpp)

ament_target_dependencies(perception_node
  rclcpp
  sensor_msgs
  crp_msgs
  std_msgs
)

install(TARGETS
  perception_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
