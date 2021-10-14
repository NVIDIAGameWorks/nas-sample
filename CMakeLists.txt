cmake_minimum_required(VERSION 3.10)

project(adaptive_shading)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_ITERATOR_DEBUG_LEVEL=1")
endif()

add_subdirectory(donut_examples)
add_subdirectory(adaptive_shading)

file(CREATE_LINK "${CMAKE_CURRENT_SOURCE_DIR}/donut_examples/media" "${CMAKE_SOURCE_DIR}/media" SYMBOLIC)
