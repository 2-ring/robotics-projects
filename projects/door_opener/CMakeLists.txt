cmake_minimum_required(VERSION 3.14)
project(door_opener)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# ── Python module ──
ament_python_install_package(door_opener)

# ── Python scripts ──
install(PROGRAMS
  scripts/sensor_node.py
  scripts/goal_classifier_node.py
  scripts/grasp_controller.py
  scripts/manipulation_planner.py
  scripts/test_manipulation.py
  scripts/rviz_visualizer.py
  DESTINATION lib/${PROJECT_NAME}
)

# ── Data directories ──
install(DIRECTORY launch  DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY config  DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY models  DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY rviz    DESTINATION share/${PROJECT_NAME}/)

ament_package()
