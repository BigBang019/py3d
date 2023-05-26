cmake_minimum_required(VERSION 3.23.0)

######### # essential to make it find nvcc@11.4 #########
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda/)

project(smooth LANGUAGES CUDA CXX)

FILE(GLOB smooth_exec "*.cu" "*.cpp")

add_executable(${PROJECT_NAME} ${smooth_exec})

target_link_libraries(${PROJECT_NAME}
    ${LIBTORCH_LIBS}
    ${CUDA_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OPENGL_LIBRARIES}
    Eigen3::Eigen
)