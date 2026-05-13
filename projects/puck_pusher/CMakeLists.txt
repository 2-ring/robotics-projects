cmake_minimum_required(VERSION 3.14)
project(puck_pusher)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# ── Python module ──
ament_python_install_package(puck_pusher)

# ── Python scripts ──
install(PROGRAMS
  scripts/sensor_node.py
  scripts/demo_replay.py
  scripts/lfd_policy.py
  scripts/test_lfd.py
  DESTINATION lib/${PROJECT_NAME}
)

# ── Data directories ──
install(DIRECTORY launch  DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY config  DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY models  DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY demos   DESTINATION share/${PROJECT_NAME}/)

ament_package()
