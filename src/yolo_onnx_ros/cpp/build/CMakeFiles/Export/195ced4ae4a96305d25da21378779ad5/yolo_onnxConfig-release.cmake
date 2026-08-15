#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "yolo_onnx::yolo_onnx" for configuration "Release"
set_property(TARGET yolo_onnx::yolo_onnx APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(yolo_onnx::yolo_onnx PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libyolo_onnx.so"
  IMPORTED_SONAME_RELEASE "libyolo_onnx.so"
  )

list(APPEND _cmake_import_check_targets yolo_onnx::yolo_onnx )
list(APPEND _cmake_import_check_files_for_yolo_onnx::yolo_onnx "${_IMPORT_PREFIX}/lib/libyolo_onnx.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
