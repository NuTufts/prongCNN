set(MODULE_NAME app)

# Set up include directories
# include_directories( ${LARCV_INCLUDE_DIR} )
# include_directories( ${LARLITE_INCLUDE_DIR} )
# include_directories( ${UBLARCVAPP_INCLUDE_DIR} )
# include_directories( ${LARFLOW_INCLUDE_DIR} )
include_directories( ${PROJECT_SOURCE_DIR} )

# for debug
#message( INFO "ublarcvapp include dir: ${UBLARCVAPP_INCLUDE_DIR}")
# message( INFO "larlite libs: ${LARLITE_LIBRARY_DIR}"  )
# message( INFO "root libraries: ${ROOT_LIBRARIES}")

# Set up library directory
# link_directories( ${LARLITE_LIBRARY_DIR} )
# link_directories( ${LARCV_LIBRARY_DIR} )
# link_directories( ${UBLARCVAPP_LIBRARY_DIR} )
# link_directories( ${LARFLOW_LIBRARY_DIR} )


# Create prepare_reco_images executable
add_executable(test_load_model test_load_model.cxx)

# Set include directories for the executable
target_include_directories(test_load_model PRIVATE
    # ${LARCV_INCLUDE_DIR}
    # ${LARLITE_INCLUDE_DIR}
    # ${UBLARCVAPP_INCLUDE_DIR}
    # ${LARFLOW_INCLUDE_DIR}
    # ${TORCH_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link libraries for the executable
target_link_libraries(test_load_model
    LArPID_Data
    LArPID_Model
#    ${ROOT_LIBRARIES}
    ${TORCH_LIBRARIES}
)

# Install executable
install(TARGETS test_load_model DESTINATION bin)
