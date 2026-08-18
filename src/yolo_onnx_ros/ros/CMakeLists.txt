cmake_minimum_required(VERSION 3.8)
project(yolo_onnx_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(geometry_msgs REQUIRED)

set(YOLO_CPP_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../cpp/build"
  CACHE PATH "Installation prefix of the yolo_onnx C++ library")

if(NOT EXISTS "${YOLO_CPP_PREFIX}/lib/libyolo_onnx.so")
  message(FATAL_ERROR
  "libyolo_onnx.so not found in '${YOLO_CPP_PREFIX}/lib'.\n"
  "Run 'make build-cpp' from the workspace root first,\n"
  "or set -DYOLO_CPP_PREFIX=<path> to the library installation prefix.")
endif()

list(PREPEND CMAKE_PREFIX_PATH "${YOLO_CPP_PREFIX}")
find_package(yolo_onnx CONFIG REQUIRED)

set(NODE_TARGET_NAME "${PROJECT_NAME}_segmentor_node")
set(COMPONENT_TARGET_NAME "${PROJECT_NAME}_segmentor_component")

# Build the ROS2 Component library
add_library(${COMPONENT_TARGET_NAME} SHARED
  src/yolo_segmentor_node.cpp
  include/yolo_onnx_ros/yolo_segmentor_node.hpp)

target_include_directories(${COMPONENT_TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${COMPONENT_TARGET_NAME}
  rclcpp
  rclcpp_components
  rclcpp_lifecycle
  sensor_msgs
  vision_msgs
  cv_bridge
  geometry_msgs
)

target_link_libraries(${COMPONENT_TARGET_NAME}
  yolo_onnx::yolo_onnx
  ${OpenCV_LIBS}
)

set_target_properties(${COMPONENT_TARGET_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN")

rclcpp_components_register_node(${COMPONENT_TARGET_NAME}
  PLUGIN "yolo_onnx_ros::YoloSegmentorNode"
    EXECUTABLE ${NODE_TARGET_NAME}
)

install(
  TARGETS ${COMPONENT_TARGET_NAME}
  EXPORT export_${COMPONENT_TARGET_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_dependencies(${COMPONENT_TARGET_NAME}
    rclcpp
    rclcpp_components
    rclcpp_lifecycle
    sensor_msgs
    vision_msgs
    cv_bridge
    geometry_msgs
)

ament_export_targets(export_${COMPONENT_TARGET_NAME} HAS_LIBRARY_TARGET)

install(
  DIRECTORY
  include/
  DESTINATION include
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Copy the shared library next to the node so the loader finds it at runtime.
install(FILES "${YOLO_CPP_PREFIX}/lib/libyolo_onnx.so"
    DESTINATION lib
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()