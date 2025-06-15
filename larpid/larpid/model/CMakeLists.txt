set(MODULE_NAME Model )

# DEFINE THE TARGET
set( LIBNAME LArPID_${MODULE_NAME} )

set( HEADERS
  TorchModel.h
)

add_library( ${LIBNAME} SHARED
  TorchModel.cxx
)

# create a property to hold our headers
set_target_properties(${LIBNAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")

target_include_directories(${LIBNAME}
  PUBLIC
  $<INSTALL_INTERFACE:include/larpid/model>
  ${TORCH_INCLUDE_DIRS}
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR} 
  ${PROJECT_SOURCE_DIR} 
)

# for debug
#message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
#message(STATUS "torch libraries: ${TORCH_LIBRARIES}")

target_link_libraries(${LIBNAME} PUBLIC
  LArPID_Data
  ${TORCH_LIBRARIES}
)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} 
  ${PROJECT_SOURCE_DIR} 
)

# install libraries and headers
install(TARGETS ${LIBNAME}
  EXPORT larpid
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/larpid/model)
