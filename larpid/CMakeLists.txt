cmake_minimum_required(VERSION 3.10)
project(larpid VERSION 1.1.0)

#---Install location by default is in the build directory
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/installed" CACHE PATH "${CMAKE_BUILD_DIR}" FORCE )
endif()

#---Module directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

## Dependencies

#---CERN ROOT
find_package(ROOT CONFIG REQUIRED)
# we set the compiler version based on the one used for ROOT
execute_process(
    COMMAND bash "-c" "root-config --features"
    OUTPUT_VARIABLE TEST_ROOT_CXX_STANDARD )
if (${TEST_ROOT_CXX_STANDARD} MATCHES "cxx17")
  message( STATUS "using cxx17")
  set(CMAKE_CXX_STANDARD 17)
elseif (${TEST_ROOT_CXX_STANDARD} MATCHES "cxx14")
  message( STATUS "using cxx14")
  set(CMAKE_CXX_STANDARD 14)
else ()
  message( STATUS "using cxx11")  
  set(CMAKE_CXX_STANDARD 11)
endif()

# #---OPENCV
# if (USE_OPENCV)
#   set(OpenCV_DIR $ENV{OPENCV_LIBDIR}/../share/OpenCV)
#   find_package(OpenCV CONFIG REQUIRED)
#   if (NOT OpenCV_FOUND)
#     set(USE_OPENCV 0)
#   endif()
# endif()

#---Eigen
#find_package(Eigen3 REQUIRED)

#---PYTHON
#find_package( PythonInterp 3 REQUIRED )
#find_package( PythonLibs 3 REQUIRED )
#execute_process( COMMAND python3 -c "import numpy; print(numpy.get_include())" OUTPUT_VARIABLE NUMPY_INCLUDES OUTPUT_STRIP_TRAILING_WHITESPACE )

#---larbys GEO2D
# use it if env variables defined
#if (DEFINED ENV{GEO2D_BASEDIR})
#  set(LARCV_USE_GEO2D 1)
#else()
#  set(LARCV_USE_GEO2D 0)
#endif()

set( larcv_DIR "$ENV{LARCV_LIBDIR}/cmake/larcv")
find_package( larcv REQUIRED )

set( larlite_DIR "$ENV{LARLITE_LIBDIR}/cmake/larlite" )
find_package( larlite REQUIRED )

# --- Torch
if (ON_FNAL)
  # For FNAL
  set( TBB_DIR "ENV{TBB_FQ_DIR}/lib" )
  find_package( TBB REQUIRED )
  set( Torch_DIR "$ENV{LIBTORCH_FQ_DIR}/share/cmake/Torch" )
  message( STATUS "Set Torch_DIR to find cmake: ${Torch_DIR}" )
else()
  set( Torch_DIR /usr/local/lib/python3.8/dist-packages/torch/share/cmake/Torch )
endif()
set(USE_CUDA OFF)
set(USE_CUDNN OFF)
set(USE_KINETO OFF)
set(USE_TENSORRT OFF)
set(TORCH_CUDA_ARCH_LIST "6.0;6.1;7.0;7.5;8.0;8.6")
#set(TORCH_CUDA_ARCH_LIST "3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0")
find_package( Torch REQUIRED )

# ## Git Submodules
# find_package(Git QUIET)
# if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
#   # Update submodules as needed
#   option(GIT_SUBMODULE "Check submodules during build" ON)
#   if(GIT_SUBMODULE)
#     message(STATUS "Submodule update")
#     execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
#       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#       RESULT_VARIABLE GIT_SUBMOD_RESULT)
#     if(NOT GIT_SUBMOD_RESULT EQUAL "0")
#       message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
#     endif()
#   endif()
# endif()

#---check json
# if(EXISTS "$ENV{NLOHMANN_JSON_DIR}/share/cmake/nlohmann_json/nlohmann_jsonConfig.cmake")
#   message( STATUS "Use cmvfs nlohmann/json")
#   set( nlohmann_json_DIR )
#   find_package( nlohmann_json REQUIRED )
#   set(LARCV_JSON_INCLUDE_DIR $ENV{NLOHMANN_JSON_INC})
# else()
#   if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/json/CMakeLists.txt")
#     message(FATAL_ERROR "The json submodule was not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
#   else()
#     ## External repos (from submodules)
#     message( STATUS "Use extern copy of json included in repo")
#     include_directories(extern/json)
#     add_subdirectory(extern/json)
#     set(LARCV_JSON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/json/single_include)
#   endif()
# endif()

## Setup

#---Define useful ROOT functions and macros (e.g. ROOT_GENERATE_DICTIONARY)
include(${ROOT_USE_FILE})

## Add Source

#---Add base of source
add_subdirectory(larpid)

#---Configuration export
include(CMakePackageConfigHelpers)
set(LARPID_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include")
set(LARPID_LIBRARY_DIR "${CMAKE_INSTALL_PREFIX}/lib")
set(TARGETS_EXPORT_NAME "larpidTargets")
configure_package_config_file("cmake/Config.cmake.in"
  "${CMAKE_BINARY_DIR}/cmake/larpidConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/larpid"
  PATH_VARS LARPID_INCLUDE_DIR LARPID_LIBRARY_DIR)
write_basic_package_version_file("${CMAKE_BINARY_DIR}/cmake/larpidConfigVersion.cmake"
  COMPATIBILITY SameMajorVersion)


## Export Targets

#---Create info to include LArPID in other cmake files
install(EXPORT larpid
        FILE larpidTargets.cmake
        DESTINATION "${CMAKE_BINARY_DIR}/installed/lib/cmake/larpid")

#---Create info to include LArPID in other cmake files
install(FILES "${CMAKE_BINARY_DIR}/cmake/larpidConfig.cmake"
  "${CMAKE_BINARY_DIR}/cmake/larpidConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/larpid")
