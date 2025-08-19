cmake_minimum_required(VERSION 3.14)
project(add_openfhe LANGUAGES CXX)

# --------------------------------------------------------------------
# 1.  Compiler / flags
# --------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# add_compile_options(-O3 -Wall -Wextra -Wno-unused-parameter)
option( BUILD_STATIC "Set to ON to include static versions of the library" OFF)

# --------------------------------------------------------------------
# 2.  Find the OpenFHE *package* that the install script put into
#     third_party/openfhe
# --------------------------------------------------------------------
#   scripts/get_openfhe.sh installs:
#     third_party/openfhe/lib/cmake/OpenFHE/OpenFHEConfig.cmake

find_package(OpenFHE CONFIG REQUIRED)
if (OpenFHE_FOUND)
    message(STATUS "FOUND PACKAGE OpenFHE")
    message(STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
    message(STATUS "OpenFHE installed as shared libraries: ${OpenFHE_SHARED}")
    message(STATUS "OpenFHE include files location: ${OpenFHE_INCLUDE}")
    message(STATUS "OpenFHE lib files location: ${OpenFHE_LIBDIR}")
    message(STATUS "OpenFHE Native Backend size: ${OpenFHE_NATIVE_SIZE}")
else()
    message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif ()

set( CMAKE_CXX_FLAGS "${OpenFHE_CXX_FLAGS} -Werror")

# --------------------------------------------------------------------
# 3.  Link libraries
# --------------------------------------------------------------------

include_directories( ${OPENMP_INCLUDES} )
include_directories( ${OpenFHE_INCLUDE} )
include_directories( ${OpenFHE_INCLUDE}/third-party/include )
include_directories( ${OpenFHE_INCLUDE}/core )
include_directories( ${OpenFHE_INCLUDE}/pke )
include_directories( ${OpenFHE_INCLUDE}/binfhe )
### add directories for other OpenFHE modules as needed for your project

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories( ${OpenFHE_LIBDIR} )
link_directories( ${OPENMP_LIBRARIES} )
link_directories( ${CMAKE_CURRENT_SOURCE_DIR}/pre-built-library )

# set(CMAKE_SKIP_BUILD_RPATH FALSE)
# set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)
# set(CMAKE_BUILD_RPATH "${OpenFHE_LIBDIR}") 
# set(CMAKE_INSTALL_RPATH "${OpenFHE_LIBDIR}") 
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_link_options(-Wl,--no-as-needed)

# These should take care of setting the correct LD path without manually setting LD_LIBRARY_PATH. 
# But if a shared library or symbol lookup error occurs, you need to set the LD_LIBRARY_PATH environment 
# to the OpenFHE lib directory: export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/path/to/OpenFHE/lib

if(BUILD_STATIC)
    set( CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries( ${OpenFHE_STATIC_LIBRARIES} )
    message(STATUS "Build with static libs")
else()
    set( CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS} )
    link_libraries( ${OpenFHE_SHARED_LIBRARIES} )
endif()

# --------------------------------------------------------------------
# 4.  Create mlp library
# --------------------------------------------------------------------

add_library( mlp_encryption_utils src/mlp_encryption_utils.cpp )

# Use pre-built mlp_openfhe library
add_library( mlp_openfhe STATIC IMPORTED )
set_target_properties( mlp_openfhe PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/pre-built-library/libmlp_openfhe.a )

# --------------------------------------------------------------------
# 5.  Each *.cpp file becomes its own executable.
#     The seven stage names are hard-wired by the benchmark contract.
# --------------------------------------------------------------------
 
add_executable( client_key_generation src/client_key_generation.cpp )

add_executable( client_preprocess_input src/client_preprocess_input.cpp )

add_executable( client_encode_encrypt_input src/client_encode_encrypt_input.cpp )
target_link_libraries( client_encode_encrypt_input mlp_encryption_utils )

add_executable( client_decrypt_decode src/client_decrypt_decode.cpp )
target_link_libraries( client_decrypt_decode mlp_encryption_utils )

add_executable( client_postprocess src/client_postprocess.cpp )

add_executable( server_preprocess_model src/server_preprocess_model.cpp )

add_executable( server_encrypted_compute src/server_encrypted_compute.cpp )
target_link_libraries( server_encrypted_compute mlp_openfhe )
target_link_libraries( server_encrypted_compute mlp_encryption_utils )
