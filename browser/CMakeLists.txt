cmake_minimum_required(VERSION 3.20)
project(dream_browser LANGUAGES CXX)
if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "Configure with emcmake cmake -S dream/browser -B build-browser")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(vm ../src)
add_executable(dream-browser
  ${vm}/value.cpp ${vm}/image.cpp ${vm}/heap.cpp ${vm}/gc_pool.cpp
  ${vm}/process.cpp ${vm}/interp.cpp ${vm}/builtins.cpp ${vm}/trunk.cpp
  ${vm}/pack.cpp ${vm}/debug.cpp ${vm}/digest.cpp ${vm}/regex.cpp
  ${vm}/bigint.cpp ${vm}/scheduler.cpp ${vm}/runtime.cpp
  ${vm}/tensor.cpp ${vm}/tensor_kernels.cpp ${vm}/gpu.cpp
  ${vm}/jit_stub.cpp ${vm}/jit_rt.cpp platform.cpp host.cpp)
target_include_directories(dream-browser PRIVATE ../src ../include)
target_compile_definitions(dream-browser PRIVATE DREAM_BROWSER=1)
target_compile_options(dream-browser PRIVATE -O2 -fexceptions)
target_link_options(dream-browser PRIVATE
  -O2 -fexceptions -sDISABLE_EXCEPTION_CATCHING=0
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=worker,node
  -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=268435456 -sSTACK_SIZE=1048576
  -sEXPORTED_FUNCTIONS=['_malloc','_free','_dream_browser_load','_dream_browser_tick','_dream_browser_delay','_dream_browser_event','_dream_browser_dispose']
  -sEXPORTED_RUNTIME_METHODS=['UTF8ToString','stringToUTF8','lengthBytesUTF8','HEAPU8'])
set_target_properties(dream-browser PROPERTIES SUFFIX .mjs)
