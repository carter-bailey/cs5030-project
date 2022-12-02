cmake_minimum_required(VERSION 3.12)

set(HEADER_FILES lib/utils.hpp lib/kmeans.hpp lib/song.hpp )

set(OPENMP openmp)

project(${OPENMP})
find_package(OpenMP REQUIRED)
add_executable(${OPENMP} openmp.cpp ${HEADER_FILES})
target_link_libraries(${OPENMP} PRIVATE OpenMP::OpenMP_CXX)
set_property(TARGET ${OPENMP} PROPERTY CXX_STANDARD 20)

set(SERIAL serial)

project(${SERIAL})
add_executable(${SERIAL} serial.cpp ${HEADER_FILES} lib/openmp.hpp)
set_property(TARGET ${SERIAL} PROPERTY CXX_STANDARD 20)

set(MPI openmpi)
project(${MPI})
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
add_executable(${MPI} openmpi.cpp ${HEADER_FILES} lib/openmpi.hpp)
set_property(TARGET ${OPENMP} PROPERTY CXX_STANDARD 20)
target_link_libraries(${OPENMP} PRIVATE MPI::MPI_CXX)


if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(${OPENMP} PRIVATE /W4 /permissive-)  
  target_compile_options(${SERIAL} PRIVATE /W4 /permissive-)  
  target_compile_options(${MPI} PRIVATE /W4 /permissive-)  
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(${OPENMP} PRIVATE -fopenmp)
  target_compile_options(${SERIAL} PRIVATE)
  target_compile_options(${MPI} PRIVATE -fopenmpi)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "CLANG")
  target_compile_options(${OPENMP} PRIVATE -fopenmp)
  target_compile_options(${SERIAL} PRIVATE)
  target_compile_options(${MPI} PRIVATE  -fopenmpi)
endif()

find_program(CLANG_FORMAT "clang-format")
if (CLANG_FORMAT)
  unset(SOURCE_FILES_PATHS)
  foreach(SOURCE_FILE ${SOURCE_FILES} ${GTEST} ${HEADER_FILES})
    get_source_file_property(WHERE ${SOURCE_FILE} LOCATION)
    set(SOURCE_FILES_PATHS ${SOURCE_FILES_PATHS} ${WHERE})
  endforeach()
  add_custom_target(ClangFormat 
    COMMAND 
    ${CLANG_FORMAT}
    -i
    -style=file
    ${SOURCE_FILES_PATHS}
    )
  add_dependencies(${OPENMP} ClangFormat)
  add_dependencies(${SERIAL} ClangFormat)
else()
  message("unable to find clang-format")
endif()
