cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0072 NEW)
cmake_policy(SET CMP0074 NEW)

project(psdr_cuda VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if( WIN32 )
    add_definitions(-D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS -DNDEBUG)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -Wall -Wno-reorder -Wno-sign-compare -fPIC")
endif()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake" ${CMAKE_MODULE_PATH})

option(DRJIT_ENABLE_JIT "" ON)
option(DRJIT_ENABLE_AUTODIFF "" ON)
option(DRJIT_ENABLE_PYTHON "" ON)

set(DRJIT_DIR ${CMAKE_SOURCE_DIR}/ext/drjit)
set(PYBIND11_ROOT ${DRJIT_DIR}/ext/pybind11)

# add_subdirectory(${DRJIT_DIR})
add_subdirectory(${PYBIND11_ROOT})

set(PSDR_CUDA_FILE psdr_jit)
add_compile_definitions(PSDR_CUDA_FILE="${PSDR_CUDA_FILE}")

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING
       "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)

option(CUDA_GENERATE_DEPENDENCIES_DURING_CONFIGURE
       "Generate dependencies during configure time instead of only during build time." OFF)

find_package(CUDA 11.0 REQUIRED)
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")


set(CUDA_GENERATED_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib/ptx")

include_directories(
    include/
    ${DRJIT_DIR}/include
    ${PYBIND11_ROOT}/include
    ${PYTHON_INCLUDE_DIRS}
    ${PYTHON_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CUDA_INCLUDE_DIRS}
)


link_directories(
    ${PYTHON_ROOT}/libs
)

# set(PYPSDR_LIBRARIES drjit-core drjit-autodiff)

set(PSDR_INCLUDE_DIR include/psdr)
set(PSDR_SOURCE_DIR src)

set(PSDR_SOURCE_FILES
    ${PSDR_INCLUDE_DIR}/psdr.h
    ${PSDR_INCLUDE_DIR}/constants.h
    ${PSDR_INCLUDE_DIR}/fwd.h
    ${PSDR_INCLUDE_DIR}/macros.h
    ${PSDR_INCLUDE_DIR}/object.h
    ${PSDR_INCLUDE_DIR}/types.h
    ${PSDR_INCLUDE_DIR}/utils.h
    ${PSDR_INCLUDE_DIR}/core/ray.h
    ${PSDR_INCLUDE_DIR}/core/frame.h
    ${PSDR_INCLUDE_DIR}/core/intersection.h
    ${PSDR_INCLUDE_DIR}/core/bitmap.h
    ${PSDR_SOURCE_DIR}/core/bitmap.cpp
    ${PSDR_INCLUDE_DIR}/core/bitmap_loader.h
    ${PSDR_SOURCE_DIR}/core/bitmap_loader.cpp
    ${PSDR_INCLUDE_DIR}/core/sampler.h
    ${PSDR_SOURCE_DIR}/core/sampler.cpp
    ${PSDR_INCLUDE_DIR}/core/records.h
    ${PSDR_INCLUDE_DIR}/core/warp.h
    ${PSDR_INCLUDE_DIR}/core/transform.h

    ${PSDR_INCLUDE_DIR}/core/cube_distrb.h
    ${PSDR_SOURCE_DIR}/core/cube_distrb.cpp


    ${PSDR_INCLUDE_DIR}/core/pmf.h
    ${PSDR_SOURCE_DIR}/core/pmf.cpp

    ${PSDR_INCLUDE_DIR}/optix/ptx.h
    ${PSDR_SOURCE_DIR}/optix/ptx.cpp


    ${PSDR_INCLUDE_DIR}/bsdf/bsdf.h




    ${PSDR_INCLUDE_DIR}/edge/edge.h

    ${PSDR_INCLUDE_DIR}/shape/mesh.h
    ${PSDR_SOURCE_DIR}/shape/mesh.cpp

    ${PSDR_INCLUDE_DIR}/emitter/emitter.h
    ${PSDR_INCLUDE_DIR}/emitter/area.h
    ${PSDR_SOURCE_DIR}/emitter/area.cpp
    ${PSDR_INCLUDE_DIR}/emitter/envmap.h
    ${PSDR_SOURCE_DIR}/emitter/envmap.cpp

    ${PSDR_INCLUDE_DIR}/sensor/sensor.h
    ${PSDR_SOURCE_DIR}/sensor/sensor.cpp

    ${PSDR_INCLUDE_DIR}/sensor/perspective.h
    ${PSDR_SOURCE_DIR}/sensor/perspective.cpp
    ${PSDR_INCLUDE_DIR}/sensor/orthographic.h
    ${PSDR_SOURCE_DIR}/sensor/orthographic.cpp

    ${PSDR_INCLUDE_DIR}/core/tinyexr.h
    ${PSDR_INCLUDE_DIR}/core/miniz.h
    ${PSDR_SOURCE_DIR}/core/miniz.cpp

    # ${PSDR_INCLUDE_DIR}/scene/optix.h
    ${PSDR_INCLUDE_DIR}/scene/scene_optix.h
    ${PSDR_SOURCE_DIR}/scene/scene_optix.cpp

    ${PSDR_INCLUDE_DIR}/scene/scene.h
    ${PSDR_SOURCE_DIR}/scene/scene.cpp

    ${PSDR_SOURCE_DIR}/optix_stubs.cpp

    ${PSDR_SOURCE_DIR}/psdr.cpp
)


set(CUDA_SOURCE_FILES
    ./cuda/kernel/util.cu
)


if( WIN32 )
    cuda_add_library(cu_library STATIC ${CUDA_SOURCE_FILES})
    cuda_add_cublas_to_target(cu_library)
    pybind11_add_module(psdr_jit  ${PSDR_SOURCE_FILES})
    target_compile_options(psdr_jit PRIVATE /O2 /wd4251 /MP)
    add_subdirectory(${DRJIT_DIR})
    set(PYPSDR_LIBRARIES drjit-core drjit-autodiff nanothread)
    target_link_libraries(psdr_jit PRIVATE cu_library ${PYPSDR_LIBRARIES} ${CUDA_LIBRARIES})
else()
    cuda_add_library(cu_library STATIC ${CUDA_SOURCE_FILES})
    cuda_add_cublas_to_target(cu_library)
    add_subdirectory(${DRJIT_DIR})
    set(PYPSDR_LIBRARIES drjit-core drjit-autodiff nanothread)
    cuda_add_library(psdr_jit MODULE ${PSDR_SOURCE_FILES})
    target_link_libraries(psdr_jit cu_library ${PYPSDR_LIBRARIES} ${CUDA_LIBRARIES})
endif()



set(PSDR_PACKAGE_DIR "${CMAKE_BINARY_DIR}/python/psdr_jit")
install(TARGETS psdr_jit drjit-core drjit-autodiff nanothread
        CONFIGURATIONS Release
        RUNTIME DESTINATION ${PSDR_PACKAGE_DIR}
        LIBRARY DESTINATION ${PSDR_PACKAGE_DIR}
)

add_custom_command(
    OUTPUT ${PSDR_PACKAGE_DIR}/__init__.py
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_SOURCE_DIR}/resources/__init__.py
    ${PSDR_PACKAGE_DIR}/__init__.py
)

add_custom_target(
    psdr-python-init-psdr_jit
    ALL DEPENDS ${PSDR_PACKAGE_DIR}/__init__.py
)

set(DRJIT_PACKAGE_DIR "${CMAKE_BINARY_DIR}/python/drjit")
install(TARGETS drjit-python drjit-core drjit-autodiff nanothread
        RUNTIME DESTINATION ${DRJIT_PACKAGE_DIR}
        LIBRARY DESTINATION ${DRJIT_PACKAGE_DIR}
)

add_custom_command(
    TARGET psdr_jit POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${DRJIT_DIR}/drjit
    ${DRJIT_PACKAGE_DIR}/
)


add_custom_target(
    TARGET psdr_jit POST_BUILD
    psdr-python-init-drjit
    ALL DEPENDS ${DRJIT_PACKAGE_DIR}/__init__.py
)

# target build
target_compile_definitions(psdr_jit PRIVATE PTX_OUTPUT_DIR="${CUDA_GENERATED_OUTPUT_DIR}")

set_target_properties(psdr_jit PROPERTIES SKIP_BUILD_RPATH FALSE)
set_target_properties(psdr_jit PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
set_property(TARGET psdr_jit PROPERTY CXX_STANDARD 17)
set_target_properties(psdr_jit PROPERTIES PREFIX "")
