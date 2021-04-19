#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "opencv_world" for configuration "Release"
set_property(TARGET opencv_world APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_world PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/x64/vc14/lib/opencv_world410.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/vc14/bin/opencv_world410.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_world )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_world "${_IMPORT_PREFIX}/x64/vc14/lib/opencv_world410.lib" "${_IMPORT_PREFIX}/x64/vc14/bin/opencv_world410.dll" )

# Import target "opencv_img_hash" for configuration "Release"
set_property(TARGET opencv_img_hash APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opencv_img_hash PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/x64/vc14/lib/opencv_img_hash410.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/x64/vc14/bin/opencv_img_hash410.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS opencv_img_hash )
list(APPEND _IMPORT_CHECK_FILES_FOR_opencv_img_hash "${_IMPORT_PREFIX}/x64/vc14/lib/opencv_img_hash410.lib" "${_IMPORT_PREFIX}/x64/vc14/bin/opencv_img_hash410.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
