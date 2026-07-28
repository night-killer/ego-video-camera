cmake_minimum_required(VERSION 3.10)
project(ego_pose_orb_slam3_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

if(NOT DEFINED ORB_SLAM3_ROOT)
  message(FATAL_ERROR "Pass -DORB_SLAM3_ROOT=/absolute/path/to/ORB_SLAM3")
endif()

list(APPEND CMAKE_MODULE_PATH "${ORB_SLAM3_ROOT}/cmake_modules")
find_package(OpenCV 4.4 REQUIRED)
find_package(Eigen3 3.1 REQUIRED)
find_package(Pangolin REQUIRED)

add_executable(orb_slam3_monocular main.cc)
target_include_directories(orb_slam3_monocular PRIVATE
  "${ORB_SLAM3_ROOT}"
  "${ORB_SLAM3_ROOT}/include"
  "${ORB_SLAM3_ROOT}/include/CameraModels"
  "${ORB_SLAM3_ROOT}/Thirdparty/Sophus"
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
)
target_link_libraries(orb_slam3_monocular PRIVATE
  "${ORB_SLAM3_ROOT}/lib/libORB_SLAM3.so"
  "${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib/libDBoW2.so"
  "${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib/libg2o.so"
  ${OpenCV_LIBS}
  ${Pangolin_LIBRARIES}
)
