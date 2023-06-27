cmake_minimum_required(VERSION 3.23.0)

######### # essential to make it find nvcc@11.4 #########
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda/)

project(py3d LANGUAGES CUDA CXX)
#
set(LIBTORCH_ROOT "${CMAKE_SOURCE_DIR}/external/libtorch")

######### # find libtorch #########
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${LIBTORCH_ROOT}")
file(GLOB LIBTORCH_LIBS "${LIBTORCH_ROOT}/lib/*.a" "${LIBTORCH_ROOT}/lib/*.so")
######### # find cudnn #########
set(CUDNN_INCLUDE_DIR /data_HDD/zhuxingyu/anaconda3/envs/p11/include)
set(CUDNN_INCLUDE_PATH /data_HDD/zhuxingyu/anaconda3/envs/p11/include)
set(CUDNN_LIB_DIR /data_HDD/zhuxingyu/anaconda3/envs/p11/lib)
set(CUDNN_PATH /data_HDD/zhuxingyu/anaconda3/envs/p11/liblibcudnn.so)
set(CUDNN_LIBRARY_PATH /data_HDD/zhuxingyu/anaconda3/envs/p11/lib)
#set(CMAKE_CXX_FLAGS "-DCGAL_USE_BASIC_VIEWER")

######### Dependencies #########
find_package(Eigen3 REQUIRED)
find_package(CUDA 11.4 REQUIRED)
find_package(Torch REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(CGAL REQUIRED)
find_package(Qt5 COMPONENTS Widgets REQUIRED)
find_package(OpenGL)

######### INCLUDE #########
include_directories(
    "${LIBTORCH_ROOT}/include/"
    "${LIBTORCH_ROOT}/include/torch/csrc/api/include/"
    ${Python_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

###### Macro ######
MACRO(SUBDIRLIST result abs_cur_dir)
    FILE(GLOB children RELATIVE ${abs_cur_dir} ${abs_cur_dir}/*)
    SET(dirlist "")
    FOREACH(child ${children})
        IF(IS_DIRECTORY ${abs_cur_dir}/${child})
            LIST(APPEND dirlist ${abs_cur_dir}/${child})
        ENDIF()
    ENDFOREACH()
    SET(${result} ${dirlist})
ENDMACRO()

######### ADD C++ #########
include_directories(src)
FILE(GLOB SRC_LIB ${CMAKE_SOURCE_DIR}/src/*/*.cu ${CMAKE_SOURCE_DIR}/src/*/*.cpp)

######### ADD Wrapper #########
SUBDIRLIST(SUBDIRS ${CMAKE_SOURCE_DIR}/wrapper)

FOREACH(SUBDIR ${SUBDIRS})
    add_subdirectory(${SUBDIR})
ENDFOREACH()