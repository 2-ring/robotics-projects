cmake_minimum_required(VERSION 3.14)
project(mujoco_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# ── Python scripts (installed as executables) ──
install(PROGRAMS
  scripts/bridge_node.py
  scripts/mujoco_sim_node.py
  scripts/turtlebot_bridge_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ── Data directories ──
install(DIRECTORY config  DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY models  DESTINATION share/${PROJECT_NAME}/)

ament_package()
