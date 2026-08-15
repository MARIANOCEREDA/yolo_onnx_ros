cmake_minimum_required(VERSION 3.22.1)
project(yolo_onnx)

find_package(OpenCV REQUIRED)


set(ONNXRUNTIME_ROOT_DIR "/usr/local" CACHE PATH "Path to ONNX Runtime installation")

include_directories("${ONNXRUNTIME_ROOT_DIR}/include"
                    "${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime")
link_directories(${ONNXRUNTIME_ROOT_DIR}/lib)

set(HEADERS
    include/yolo_onnx/YoloOnnx.hpp
)

set(SOURCES
    src/YoloOnnx.cpp
)

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

add_executable(inference_test src/main.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} onnxruntime)

target_link_libraries(inference_test PRIVATE ${PROJECT_NAME} ${OpenCV_LIBS})

target_include_directories(inference_test PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Config
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)

install(EXPORT ${PROJECT_NAME}Config
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake/${PROJECT_NAME}
)