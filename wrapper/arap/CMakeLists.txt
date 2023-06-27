project(arap LANGUAGES CUDA CXX)

FILE(GLOB ${PROJECT_NAME}_exec "*.cu" "*.cpp")

add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_exec})
target_link_libraries(${PROJECT_NAME}
    ${LIBTORCH_LIBS}
    ${CUDA_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OPENGL_LIBRARIES}
    Eigen3::Eigen
)