set(MODULE_NAME DataPrep)

# Set up include directories
include_directories( ${LARCV_INCLUDE_DIR} )
include_directories( ${LARLITE_INCLUDE_DIR} )
include_directories( ${UBLARCVAPP_INCLUDE_DIR} )
include_directories( ${LARFLOW_INCLUDE_DIR} )
include_directories( ${PROJECT_SOURCE_DIR} )

# for debug
#message( INFO "ublarcvapp include dir: ${UBLARCVAPP_INCLUDE_DIR}")
message( INFO "larlite libs: ${LARLITE_LIBRARY_DIR}"  )

# Set up library directory
link_directories( ${LARLITE_LIBRARY_DIR} )
link_directories( ${LARCV_LIBRARY_DIR} )
link_directories( ${UBLARCVAPP_LIBRARY_DIR} )
link_directories( ${LARFLOW_LIBRARY_DIR} )


# Create prepare_reco_images executable
add_executable(prepare_reco_images prepare_reco_images.cxx prepare_reco_images.h)

# Set include directories for the executable
target_include_directories(prepare_reco_images PRIVATE
    ${LARCV_INCLUDE_DIR}
    ${LARLITE_INCLUDE_DIR}
    ${UBLARCVAPP_INCLUDE_DIR}
    ${LARFLOW_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link libraries for the executable
target_link_libraries(prepare_reco_images
    ${ROOT_LIBRARIES}
    larlite_Base
    larlite_DataFormat
    larlite_LArUtil
    LArCVCoreBase
    LArCVCoreDataFormat
    LArCVApp_MCTools
    LArFlow_PrepFlowMatchData
)

# Install executable
install(TARGETS prepare_reco_images DESTINATION bin)
