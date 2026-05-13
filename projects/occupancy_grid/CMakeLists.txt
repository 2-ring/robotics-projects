cmake_minimum_required(VERSION 3.14)
project(occupancy_grid)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# ── Python module ──
ament_python_install_package(occupancy_grid)

# ── Python scripts ──
install(PROGRAMS
  scripts/rviz_marker_node.py
  scripts/occupancy_mapper_node.py
  scripts/test_occupancy_mapper.py
  scripts/pure_pursuit_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ── Data directories ──
install(DIRECTORY launch  DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY config  DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY models  DESTINATION share/${PROJECT_NAME}/)

ament_package()
