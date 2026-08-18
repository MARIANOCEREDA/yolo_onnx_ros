cmake_minimum_required(VERSION 3.22.1)
project(yolo_onnx)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "Default install prefix" FORCE)
endif()

find_package(OpenCV REQUIRED)


set(ONNXRUNTIME_ROOT_DIR "/usr/local" CACHE PATH "Path to ONNX Runtime installation")

include_directories("${ONNXRUNTIME_ROOT_DIR}/include"
                    "${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime")
link_directories(${ONNXRUNTIME_ROOT_DIR}/lib)

set(HEADERS
    include/yolo_onnx/YoloORTInterface.hpp
    include/yolo_onnx/YoloORTDetector.hpp
    include/yolo_onnx/YoloUtils.hpp
    include/yolo_onnx/YoloORTSegmentor.hpp
)

set(SOURCES
    src/YoloORTInterface.cpp
    src/YoloORTDetector.cpp
    src/YoloUtils.cpp
    src/YoloORTSegmentor.cpp
)

# Target library
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} onnxruntime)

# Target inference test
add_executable(inference_test src/main.cpp)
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

# --- Valgrind memcheck target ---
find_program(VALGRIND valgrind)
if(VALGRIND)
    add_custom_target(valgrind
        COMMAND ${VALGRIND}
            --leak-check=full
            --show-leak-kinds=no
            --track-origins=yes
            --error-exitcode=1
            $<TARGET_FILE:inference_test>
        DEPENDS inference_test
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running Valgrind memcheck on inference_test"
        VERBATIM
    )
else()
    message(WARNING "Valgrind not found — 'valgrind' target will not be available")
endif()