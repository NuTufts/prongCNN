set(MODULE_NAME Data )

# DEFINE THE TARGET
set( LIBNAME LArPID_${MODULE_NAME} )

set( HEADERS
  ModelOutput.h
  CropPixData_t.h
  NuVertexCandidateHDF5Reader.h
)

set( ROOTDICT_HEADERS ${HEADERS} )

add_library( ${LIBNAME} SHARED
  ModelOutput.cxx
  CropPixData_t.cxx
  NuVertexCandidateHDF5Reader.cxx
)

# create a property to hold our headers
set_target_properties(${LIBNAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")

target_include_directories(${LIBNAME}
  PUBLIC
  $<INSTALL_INTERFACE:include/larpid/data>
  ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(${LIBNAME} PUBLIC
  ${ROOT_LIBRARIES}
  ${HDF5_LIBRARIES} 
  ${HDF5_HL_LIBRARIES}
)
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} 
  ${PROJECT_SOURCE_DIR}
  ${HDF5_INCLUDE_DIRS}
)
ROOT_GENERATE_DICTIONARY(G__${LIBNAME} ${ROOTDICT_HEADERS} LINKDEF LinkDef.h)
target_sources( ${LIBNAME} PRIVATE G__${LIBNAME}.cxx )

# install libraries and headers
install(TARGETS ${LIBNAME}
  EXPORT larpid
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/larpid/data)

# install files needed for dictionary use in CINT/CLING/PYTHON
install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lib${LIBNAME}_rdict.pcm
  ${CMAKE_CURRENT_BINARY_DIR}/lib${LIBNAME}.rootmap
  DESTINATION lib  )
