cmake_minimum_required(VERSION 3.8)
project(system_identification)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(system_identification_node
  src/system_identification_node.cpp
)

target_include_directories(system_identification_node PUBLIC
  include
)

target_link_libraries(system_identification_node PUBLIC
  Eigen3::Eigen
)

ament_target_dependencies(system_identification_node
  PUBLIC
  rclcpp
  nav_msgs
  geometry_msgs
)

install(TARGETS
  system_identification_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()