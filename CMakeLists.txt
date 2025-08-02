cmake_minimum_required(VERSION 3.15.0)
project(KVLib LANGUAGES CUDA CXX C)

# find python
find_package(Python 3.6 COMPONENTS Interpreter Development REQUIRED)
set(PYTHON_INTERP python)
message(STATUS "Using Python interpreter: ${PYTHON_INTERP}")

# find CUDA
find_package(CUDAToolkit REQUIRED)
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_ARCH)
set(TORCH_CUDA_ARCH_LIST "${INSTALLED_GPU_ARCH}")
message("Detected CUDA architectures: ${TORCH_CUDA_ARCH_LIST}")

# find torch
file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/scripts/find_torch.py FIND_TORCH_PY)
execute_process(
  COMMAND ${PYTHON_INTERP} ${FIND_TORCH_PY}
  OUTPUT_VARIABLE TORCH_PREFIX_VER
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "find_torch.py output: ${TORCH_PREFIX_VER}")
list(GET TORCH_PREFIX_VER 0 TORCH_PREFIX)
list(GET TORCH_PREFIX_VER 1 TORCH_VER)
message(STATUS "Configuring for PyTorch ${TORCH_VER}")
set(Torch_DIR "${TORCH_PREFIX}/Torch")
find_package(Torch REQUIRED)
# link against libtorch_python.so
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
list(APPEND KVLib_EXTERNAL_LIB ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

# find pybind11
file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/scripts/find_pybind11.py FIND_PYBIND_PY)
execute_process(
  COMMAND ${PYTHON_INTERP} ${FIND_PYBIND_PY}
  OUTPUT_VARIABLE PYBIND_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "find_pybind11.py output: ${PYBIND_CMAKE_DIR}")
set(pybind11_DIR ${PYBIND_CMAKE_DIR})
find_package(pybind11 REQUIRED)

# set flag
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp")

# Set openmp
include(FindOpenMP)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
endif(OPENMP_FOUND)
message(STATUS "Build with OpenMP")

# Define our library target
file(GLOB KVLib_SRC
  src/*.c
  src/*.cc
  src/*.cu
  src/cuda-attn/*.cu
  src/cpu-attn/*.cc
)
pybind11_add_module(KVLib ${KVLib_SRC})
target_include_directories(KVLib PUBLIC 3rdparty/raft/cpp/include)
target_include_directories(KVLib PUBLIC 3rdparty/rmm/include)
target_include_directories(KVLib PUBLIC 3rdparty/spdlog/include)
target_include_directories(KVLib PUBLIC 3rdparty/cutlass/include)
set_target_properties(KVLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(KVLib PROPERTIES CUDA_STANDARD 17)
set_target_properties(KVLib PROPERTIES POSITION_INDEPENDENT_CODE ON)
TARGET_COMPILE_OPTIONS(
  KVLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
target_compile_options(KVLib PRIVATE -mavx512f)
target_compile_options(KVLib PRIVATE -mf16c)

# Enable C++17
target_compile_features(KVLib PRIVATE cxx_std_17)

# Link against LibTorch
target_link_libraries(KVLib PRIVATE ${KVLib_EXTERNAL_LIB})
target_link_libraries(KVLib PRIVATE cuda)
