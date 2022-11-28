cmake_minimum_required(VERSION 3.12)

set(SOURCE_FILES main.cpp )
set(HEADER_FILES utils.hpp knn.hpp song.hpp )
set(PROJECTNAME knn)

project(${PROJECTNAME})
# randDistributions
add_executable(${PROJECTNAME} ${SOURCE_FILES} ${HEADER_FILES})
set_property(TARGET ${PROJECTNAME} PROPERTY CXX_STANDARD 20)

set_property(TARGET ${GTESTNAME} PROPERTY CXX_STANDARD 20)


if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(${PROJECTNAME} PRIVATE /W4 /permissive-)  
  set_target_properties(${PROJECTNAME} PROPERTIES LINK_FLAGS "/STACK:10000000")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(${PROJECTNAME} PRIVATE -Wall -Wextra -pedantic -O3)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "CLANG")
  target_compile_options(${PROJECTNAME} PRIVATE -Wall -Wextra -pedantic -O3 -fopenmp=libomp)
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
  add_dependencies(${PROJECTNAME} ClangFormat)
else()
  message("unable to find clang-format")
endif()
