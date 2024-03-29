cmake_minimum_required(VERSION 3.15...3.22)

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if( WIN32 )
    add_definitions(-D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS -DNDEBUG)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -Wall -Wno-reorder -Wno-sign-compare -fPIC")
endif()

option(DRJIT_ENABLE_JIT "" ON)
option(DRJIT_ENABLE_AUTODIFF "" ON)
option(DRJIT_ENABLE_PYTHON "" ON)

set(DRJIT_DIR ${CMAKE_SOURCE_DIR}/ext/drjit)
set(PYBIND11_ROOT ${DRJIT_DIR}/ext/pybind11)

add_subdirectory(${PYBIND11_ROOT})

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

option(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING
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
    ${PSDR_INCLUDE_DIR}/bsdf/diffuse.h
    ${PSDR_SOURCE_DIR}/bsdf/diffuse.cpp

    ${PSDR_INCLUDE_DIR}/bsdf/microfacet.h
    ${PSDR_SOURCE_DIR}/bsdf/microfacet.cpp

    ${PSDR_INCLUDE_DIR}/bsdf/microfacet_pv.h
    ${PSDR_SOURCE_DIR}/bsdf/microfacet_pv.cpp

    ${PSDR_INCLUDE_DIR}/bsdf/roughconductor.h
    ${PSDR_SOURCE_DIR}/bsdf/roughconductor.cpp
    ${PSDR_INCLUDE_DIR}/bsdf/roughdielectric.h
    ${PSDR_SOURCE_DIR}/bsdf/roughdielectric.cpp

    ${PSDR_INCLUDE_DIR}/bsdf/normalmap.h
    ${PSDR_SOURCE_DIR}/bsdf/normalmap.cpp

    ${PSDR_INCLUDE_DIR}/bsdf/ggx.h
    ${PSDR_SOURCE_DIR}/bsdf/ggx.cpp

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

    ${PSDR_INCLUDE_DIR}/scene/scene_optix.h
    ${PSDR_SOURCE_DIR}/scene/scene_optix.cpp

    ${PSDR_INCLUDE_DIR}/scene/scene.h
    ${PSDR_SOURCE_DIR}/scene/scene.cpp

    ${PSDR_INCLUDE_DIR}/scene/scene_loader.h
    ${PSDR_SOURCE_DIR}/scene/scene_loader.cpp

    ${PSDR_INCLUDE_DIR}/integrator/integrator.h
    ${PSDR_SOURCE_DIR}/integrator/integrator.cpp
    ${PSDR_INCLUDE_DIR}/integrator/field.h
    ${PSDR_SOURCE_DIR}/integrator/field.cpp

    ${PSDR_INCLUDE_DIR}/integrator/collocated.h
    ${PSDR_SOURCE_DIR}/integrator/collocated.cpp

    ${PSDR_INCLUDE_DIR}/integrator/direct.h
    ${PSDR_SOURCE_DIR}/integrator/direct.cpp


    ${PSDR_INCLUDE_DIR}/integrator/path.h
    ${PSDR_SOURCE_DIR}/integrator/path.cpp

    ${PSDR_SOURCE_DIR}/jit_optix_test.cpp

    ${PSDR_SOURCE_DIR}/optix_stubs.cpp

    ${PSDR_SOURCE_DIR}/psdr.cpp
)

set(CUDA_SOURCE_FILES
    # ./cuda/kernel/knn.cu
    ./cuda/kernel/util.cu
    # ./cuda/kernel/NEE.cu
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

install(TARGETS psdr_jit drjit-core drjit-autodiff nanothread DESTINATION psdr_jit)
install(TARGETS drjit-python drjit-core drjit-autodiff nanothread DESTINATION drjit)
install(DIRECTORY ${DRJIT_DIR}/drjit DESTINATION ${SKBUILD_PLATLIB_DIR})

target_compile_definitions(psdr_jit PRIVATE PTX_OUTPUT_DIR="${CUDA_GENERATED_OUTPUT_DIR}")
set_target_properties(psdr_jit drjit-python drjit-core drjit-autodiff nanothread PROPERTIES
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH "$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH TRUE)
set_property(TARGET psdr_jit PROPERTY CXX_STANDARD 17)
set_target_properties(psdr_jit PROPERTIES PREFIX "")