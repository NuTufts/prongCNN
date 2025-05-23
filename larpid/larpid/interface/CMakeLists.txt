set(MODULE_NAME Interface )

# DEFINE THE TARGET
set( LIBNAME LArPID_${MODULE_NAME} )

set( HEADERS
  ModelOutput.h
  LArPIDInterface.h
  TorchModel.h
)

set( ROOTDICT_HEADERS ${HEADERS} )
list( FILTER ROOTDICT_HEADERS EXCLUDE REGEX "TorchModel.h" )


add_library( ${LIBNAME} SHARED
  ModelOutput.cxx
  LArPIDInterface.cxx
  TorchModel.cxx
)

# create a property to hold our headers
set_target_properties(${LIBNAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")

# DEFINE SOME VARS THAT MIGHT BE USEFUL FOR SUBMODULES
#message("python include: ${PYTHON_INCLUDE_PATH}")
#get_target_property(xgboost_include_dirs xgboost::xgboost INTERFACE_INCLUDE_DIRECTORIES)
#message("xgboost include: ${xgboost_include_dirs}")

# set( PUB_INC_DIRS  ${LARCV_INCLUDE_DIR}
#   ${UBLARCVAPP_INCLUDE_DIR}
#   ${LARLITE_INC_DIRS}
#   ${PYTHON_INCLUDE_PATH}
#   ${NUMPY_INCLUDES} )
# set( PRI_INC_DIRS
#   ${PROJECT_SOURCE_DIR}
#   ${EIGEN3_INCLUDE_DIRS}
#   ${CILANTRO_INC_DIR}
#   ${OpenCV_INCLUDE_DIRS} )
# set( PRIVATE_LIBS_USED "" )
# if(xgboost_FOUND)
#   set( PRI_INC_DIRS ${PRI_INC_DIRS} ${xgboost_include_dirs} )
#   set( PRIVATE_LIBS_USED xgboost::xgboost )
# endif()

# larcv libs used
#set(LARCV_LIBS_USED LArCVCoreBase LArCVCoreDataFormat LArCVCoreProcessor LArCVApp_dbscan LArCVApp_MCTools )
#
# ublarcvapp libs used
#set(UBLARCVAPP_LIBS_USED LArCVApp_UBImageMod)

# larflow libs used
#set(LARFLOW_LIBS_USED LArFlow_RecoUtils)

message( STATUS "larcv ${LARCV_INCLUDE_DIR}")

target_include_directories(${LIBNAME}
  PUBLIC
  $<INSTALL_INTERFACE:include/larpid/${MODULE_NAME}>
  ${TORCH_INCLUDE_DIRS}
  ${LARLITE_INCLUDE_DIR}
  ${LARCV_INCLUDE_DIR}
)

message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")

target_link_libraries(${LIBNAME} PUBLIC
  larlite_DataFormat
  larlite_LArUtil
  LArCVCoreBase
  LArCVCoreDataFormat
  ${TORCH_LIBRARIES}
)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR} ${LARCV_INCLUDE_DIR} )
# ${LARLITE_INC_DIRS} ${PYTHON_INCLUDE_PATH} ${NUMPY_INCLUDES} ${OpenCV_INCLUDE_DIRS} ${xgboost_include_dirs} )
ROOT_GENERATE_DICTIONARY(G__${LIBNAME} ${ROOTDICT_HEADERS} LINKDEF LinkDef.h)
target_sources( ${LIBNAME} PRIVATE G__${LIBNAME}.cxx )

# install libraries and headers
install(TARGETS ${LIBNAME}
  EXPORT larpid
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/larpid/${MODULE_NAME})

# install files needed for dictionary use in CINT/CLING/PYTHON
install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lib${LIBNAME}_rdict.pcm
  ${CMAKE_CURRENT_BINARY_DIR}/lib${LIBNAME}.rootmap
  DESTINATION lib  )
