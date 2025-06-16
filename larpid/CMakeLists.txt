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

# --- HDF5
find_package(HDF5 REQUIRED COMPONENTS C CXX HL)

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

set( ublarcvapp_DIR "$ENV{UBLARCVAPP_LIBDIR}/cmake/ublarcvapp" )
find_package( ublarcvapp REQUIRED )

# --- Torch
if (ON_FNAL)
  # For FNAL
  set( TBB_DIR "ENV{TBB_FQ_DIR}/lib" )
  find_package( TBB REQUIRED )
  set( Torch_DIR "$ENV{LIBTORCH_FQ_DIR}/share/cmake/Torch" )
  message( STATUS "Set Torch_DIR to find cmake: ${Torch_DIR}" )
else()
  # Need special torch built with cxx11-ABI
  #set( Torch_DIR /usr/local/lib/python3.8/dist-packages/torch/share/cmake/Torch )
  set( Torch_DIR "$ENV{LIBTORCH_DIR}/share/cmake/Torch" )
endif()
set(USE_CUDA OFF)
set(USE_CUDNN OFF)
set(USE_KINETO OFF)
set(USE_TENSORRT OFF)
set(TORCH_CUDA_ARCH_LIST "6.0;6.1;7.0;7.5;8.0;8.6")
#set(TORCH_CUDA_ARCH_LIST "3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0")
find_package( Torch REQUIRED )

# Check CXX11 ABI compatibility
# We'll check if the torch libraries contain the expected ABI symbols
# This avoids the complexity of trying to compile/link during cmake configuration

# First check what ABI the compiler is using
file(WRITE ${CMAKE_BINARY_DIR}/test_compiler_abi.cpp
"#include <iostream>
int main() {
#ifdef _GLIBCXX_USE_CXX11_ABI
    std::cout << _GLIBCXX_USE_CXX11_ABI;
#else
    std::cout << \"0\";
#endif
    return 0;
}
")

try_run(
    COMPILER_ABI_RUN_RESULT
    COMPILER_ABI_COMPILE_RESULT
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/test_compiler_abi.cpp
    RUN_OUTPUT_VARIABLE COMPILER_ABI_VALUE
)

if(COMPILER_ABI_COMPILE_RESULT)
    message(STATUS "Compiler is using _GLIBCXX_USE_CXX11_ABI=${COMPILER_ABI_VALUE}")
else()
    message(WARNING "Could not determine compiler ABI setting")
    set(COMPILER_ABI_VALUE "1")  # Assume CXX11 ABI since that's what we need
endif()

# Now check LibTorch libraries for ABI compatibility
# Look for the torch library file - check multiple possible locations
set(TORCH_LIB_LOCATION "")
set(POSSIBLE_TORCH_LIBS 
    "${TORCH_INSTALL_PREFIX}/lib/libtorch_python.so"
    "${TORCH_LIBRARIES}"
    "${CMAKE_PREFIX_PATH}/lib/libtorch_python.so"
    "$ENV{LIBTORCH_DIR}/lib/libtorch_python.so"
)

foreach(LIB ${POSSIBLE_TORCH_LIBS})
    if(EXISTS ${LIB})
        set(TORCH_LIB_LOCATION ${LIB})
        break()
    endif()
endforeach()

if(NOT EXISTS ${TORCH_LIB_LOCATION})
    # Try to get it from the torch target
    get_target_property(TORCH_LIB_LOCATION torch LOCATION)
endif()

message(STATUS "Checking LibTorch library at: ${TORCH_LIB_LOCATION}")

# Use strings command to check for CXX11 ABI symbols
find_program(STRINGS_EXECUTABLE strings)

set(TORCH_ABI_DETECTED "unknown")

if(STRINGS_EXECUTABLE AND EXISTS ${TORCH_LIB_LOCATION})
    # Look for cxx1112basic_string pattern which indicates CXX11 ABI
    execute_process(
        COMMAND ${STRINGS_EXECUTABLE} ${TORCH_LIB_LOCATION}
        COMMAND grep "cxx1112basic_string"
        COMMAND head -n 1
        OUTPUT_VARIABLE STRINGS_OUTPUT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(STRINGS_OUTPUT)
        set(TORCH_ABI_DETECTED "1")
        message(STATUS "Detected LibTorch was built with CXX11 ABI (found cxx1112basic_string symbols)")
    else()
        # Double check by looking for the __cxx11 namespace
        execute_process(
            COMMAND ${STRINGS_EXECUTABLE} ${TORCH_LIB_LOCATION}
            COMMAND grep "__cxx11"
            COMMAND head -n 1
            OUTPUT_VARIABLE STRINGS_OUTPUT2
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        if(STRINGS_OUTPUT2)
            set(TORCH_ABI_DETECTED "1")
            message(STATUS "Detected LibTorch was built with CXX11 ABI (found __cxx11 namespace)")
        else()
            set(TORCH_ABI_DETECTED "0")
            message(STATUS "LibTorch appears to be built with old ABI (no CXX11 markers found)")
        endif()
    endif()
else()
    message(WARNING "Could not find 'strings' command to check LibTorch ABI. Build may fail if ABI is mismatched.")
endif()

# Check if ABIs match
if(COMPILER_ABI_VALUE STREQUAL "1" AND TORCH_ABI_DETECTED STREQUAL "0")
    message(FATAL_ERROR "\n\nLibTorch ABI Compatibility Error:\n"
        "Your compiler/dependencies use _GLIBCXX_USE_CXX11_ABI=1,\n"
        "but LibTorch was built with _GLIBCXX_USE_CXX11_ABI=0.\n\n"
        "Please install a CXX11 ABI version of LibTorch.\n"
        "You can download pre-compiled versions from:\n"
        "https://download.pytorch.org/libtorch/cu111 (or appropriate CUDA version)\n"
        "Look for versions with 'cxx11-abi' in the filename.\n\n"
        "Set the environment variable LIBTORCH_DIR to point to the CXX11 ABI version.")
elseif(COMPILER_ABI_VALUE STREQUAL "0" AND TORCH_ABI_DETECTED STREQUAL "1")
    message(FATAL_ERROR "\n\nLibTorch ABI Compatibility Error:\n"
        "Your compiler/dependencies use _GLIBCXX_USE_CXX11_ABI=0,\n"
        "but LibTorch was built with _GLIBCXX_USE_CXX11_ABI=1.\n\n"
        "You need to either:\n"
        "1. Install a non-CXX11 ABI version of LibTorch, or\n"
        "2. Rebuild your dependencies with CXX11 ABI enabled.")
elseif(TORCH_ABI_DETECTED STREQUAL "unknown")
    message(WARNING "\n\nLibTorch ABI Compatibility Warning:\n"
        "Could not detect LibTorch ABI version.\n"
        "Build may fail if there is an ABI mismatch.\n\n"
        "If you encounter linking errors, ensure LibTorch matches your compiler's ABI:\n"
        "Your compiler uses _GLIBCXX_USE_CXX11_ABI=${COMPILER_ABI_VALUE}")
else()
    message(STATUS "LibTorch ABI compatibility check passed (both using CXX11_ABI=${COMPILER_ABI_VALUE})")
endif()

# Set up RPATH for executables to find PyTorch libraries
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Get the directory containing PyTorch libraries
get_target_property(TORCH_LIBRARY_DIR torch LOCATION)
get_filename_component(TORCH_LIBRARY_DIR ${TORCH_LIBRARY_DIR} DIRECTORY)
message(STATUS "PyTorch library directory: ${TORCH_LIBRARY_DIR}")

# Set RPATH for installed executables
list(APPEND CMAKE_INSTALL_RPATH ${TORCH_LIBRARY_DIR})
list(APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

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
