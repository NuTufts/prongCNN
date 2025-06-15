set(MODULE_NAME Interface )

# DEFINE THE TARGET
set( LIBNAME LArPID_${MODULE_NAME} )

set( HEADERS
  LArPIDInterface.h
)

set( ROOTDICT_HEADERS ${HEADERS} )
list( FILTER ROOTDICT_HEADERS EXCLUDE REGEX "TorchModel.h" )


add_library( ${LIBNAME} SHARED
  LArPIDInterface.cxx
)

# create a property to hold our headers
set_target_properties(${LIBNAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")

message( STATUS "larcv ${LARCV_INCLUDE_DIR}")

target_include_directories(${LIBNAME}
  PUBLIC
  $<INSTALL_INTERFACE:include/larpid/interface>
  ${LARLITE_INCLUDE_DIR}
  ${LARCV_INCLUDE_DIR}
  ${LARFLOW_INCLUDE_DIR}
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR} 
  ${PROJECT_SOURCE_DIR} 
)

message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")

target_link_libraries(${LIBNAME} PUBLIC
  LArPID_Data
  larlite_DataFormat
  larlite_LArUtil
  larlite_Base
  LArCVCoreBase
  LArCVCoreDataFormat
  ${TORCH_LIBRARIES}
  ${ROOT_LIBRARIES}
)
message(STATUS "torch libraries: ${TORCH_LIBRARIES}")

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} 
  ${PROJECT_SOURCE_DIR} 
  ${LARCV_INCLUDE_DIR} 
  ${LARFLOW_INCLUDE_DIR}
)
# ${LARLITE_INC_DIRS} ${PYTHON_INCLUDE_PATH} ${NUMPY_INCLUDES} ${OpenCV_INCLUDE_DIRS} ${xgboost_include_dirs} )
ROOT_GENERATE_DICTIONARY(G__${LIBNAME} ${ROOTDICT_HEADERS} LINKDEF LinkDef.h)
target_sources( ${LIBNAME} PRIVATE G__${LIBNAME}.cxx )

# Create prepare_reco_images executable
add_executable(test_interface test_interface.cxx)


# Link libraries for the executable
target_link_libraries(test_interface
PUBLIC
  LArPID_Data
  LArPID_Interface
  LArFlow_Reco
  LArCVCoreBase 
  LArCVCoreDataFormat
  LArCVCoreProcessor
  ${ROOT_LIBRARIES}
)

# install libraries and headers
install(TARGETS ${LIBNAME}
  EXPORT larpid
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/larpid/interface)

# install files needed for dictionary use in CINT/CLING/PYTHON
install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lib${LIBNAME}_rdict.pcm
  ${CMAKE_CURRENT_BINARY_DIR}/lib${LIBNAME}.rootmap
  DESTINATION lib  )

# Install executable
install(TARGETS test_interface DESTINATION bin)
