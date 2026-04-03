cmake_minimum_required(VERSION 3.20)
project(medpal_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler optimizations for Raspberry Pi 5 (ARM Cortex-A76)
add_compile_options(-O3)
add_compile_options(-march=armv8-a+fp+simd+crc+crypto)
add_compile_options(-mtune=cortex-a76)
add_compile_options(-ffast-math)
add_compile_options(-flto)
add_compile_options(-Wall -Wextra)

# Find OpenCV
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc dnn objdetect highgui videoio)

message(STATUS "OpenCV found: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# Orbbec SDK (optional)
set(OB_SDK_DIR "/home/medpal/pyorbbecsdk/sdk" CACHE PATH "Orbbec SDK path")
if(EXISTS ${OB_SDK_DIR}/include)
    message(STATUS "Orbbec SDK found at ${OB_SDK_DIR}")
    include_directories(${OB_SDK_DIR}/include)
    link_directories(${OB_SDK_DIR}/lib/arm64)
    add_definitions(-DHAVE_ORBBEC=1)
endif()

# Coral EdgeTPU (optional - auto-detected at runtime)
find_library(EDGETPU_LIB edgetpu HINTS /usr/lib /usr/local/lib /opt/lib)
if(EDGETPU_LIB)
    message(STATUS "Coral EdgeTPU runtime found")
    add_definitions(-DHAVE_EDGETPU=1)
endif()

# Main executable
add_executable(vision_server
    src/main.cpp
)

target_link_libraries(vision_server
    opencv_core opencv_imgproc opencv_dnn opencv_objdetect opencv_highgui opencv_videoio
    pthread
)

# Link Orbbec if available
if(EXISTS ${OB_SDK_DIR}/lib/arm64/libObsensorSDK.so)
    target_link_libraries(vision_server ObsensorSDK)
elseif(EXISTS ${OB_SDK_DIR}/lib/arm64/libobsensor.so)
    target_link_libraries(vision_server obsensor)
endif()

# Link Coral if available
if(EDGETPU_LIB)
    target_link_libraries(vision_server ${EDGETPU_LIB})
endif()

# Install
install(TARGETS vision_server DESTINATION bin)
install(DIRECTORY models/ DESTINATION share/medpal/models)

# Print configuration summary
message(STATUS "")
message(STATUS "=== MedPal Vision Build Configuration ===")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "Orbbec SDK: ${OB_SDK_DIR}")
message(STATUS "Coral EdgeTPU: ${EDGETPU_LIB}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "========================================")
