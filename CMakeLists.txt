cmake_minimum_required(VERSION 3.14)
project(rinex_snr_extractor C)

set(CMAKE_C_STANDARD 99)
set(RTKLIB_SRC_DIR ${CMAKE_SOURCE_DIR}/third_party/rtklib/src)
include_directories(${CMAKE_SOURCE_DIR}/third_party/rtklib/src)
# 有的版本把头文件放 include/，按需加：
include_directories(${CMAKE_SOURCE_DIR}/third_party/rtklib/include)

add_executable(snr_extract
    src/snr_extract.c
    src/signals.h
    ${RTKLIB_SRC_DIR}/rtkcmn.c
    ${RTKLIB_SRC_DIR}/rinex.c
    ${RTKLIB_SRC_DIR}/preceph.c
)

# 一些平台需要数学库
if(UNIX OR MINGW)
  target_link_libraries(snr_extract m)
endif()
