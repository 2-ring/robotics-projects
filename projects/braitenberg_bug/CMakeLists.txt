cmake_minimum_required(VERSION 3.14)
project(braitenberg_bug)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# ── Python scripts ──
install(PROGRAMS
  scripts/sensor_node.py
  scripts/braitenberg_controller.py
  scripts/test_braitenberg.py
  DESTINATION lib/${PROJECT_NAME}
)

# ── Data directories ──
install(DIRECTORY launch  DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY config  DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY models  DESTINATION share/${PROJECT_NAME}/)

ament_package()
