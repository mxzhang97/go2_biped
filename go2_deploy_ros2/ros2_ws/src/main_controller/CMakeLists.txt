cmake_minimum_required(VERSION 3.8)
project(main_controller)


if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_PREFIX_PATH /home/max/libtorch) 

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(unitree_go REQUIRED)
find_package(unitree_api REQUIRED)
find_package(Torch REQUIRED)


if(TORCH_CUDA_FOUND)
  message(STATUS "LibTorch CUDA support found")
else()
  message(STATUS "LibTorch CPU only")
endif()

include_directories(
  include
  include/common
  ${TORCH_INCLUDE_DIRS}
)

add_executable(main_policy_node
src/main_controller.cpp
src/common/motor_crc.cpp
)


ament_target_dependencies(main_policy_node
  rclcpp
  std_msgs
  unitree_go
  unitree_api
)

target_link_libraries(main_policy_node
  ${TORCH_LIBRARIES}
)

install(TARGETS main_policy_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)




if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
