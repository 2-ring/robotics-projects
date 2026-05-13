cmake_minimum_required(VERSION 3.14)
project(panda_teleop)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# ── Python scripts (installed as executables) ──
install(PROGRAMS
  scripts/joint_pose_commander.py
  scripts/keyboard_teleop.py
  DESTINATION lib/${PROJECT_NAME}
)

# ── Data directories ──
install(DIRECTORY launch  DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY config  DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY models  DESTINATION share/${PROJECT_NAME}/)

ament_package()
