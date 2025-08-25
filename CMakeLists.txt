cmake_minimum_required(VERSION 3.20)
project(rinex_snr_extractor C CXX)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

# === 可选：如果你的 RTKLIB 使用小写 snr 字段，请打开下一行 ===
# add_compile_definitions(SNR_FIELD=snr)

# RTKLIB
set(RTKLIB_DIR "${CMAKE_SOURCE_DIR}/third_party/rtklib" CACHE PATH "Path to RTKLIB root")
set(RTKLIB_SRC "${RTKLIB_DIR}/src")
include_directories(${RTKLIB_SRC})
if (EXISTS "${RTKLIB_DIR}/include")
  include_directories(${RTKLIB_DIR}/include)
endif()

# Windows 兼容项
if (MSVC)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS strtok_r=strtok_s)
  add_compile_options(/utf-8)
endif()
if (MINGW)
  add_compile_definitions(strtok_r=strtok_s)
endif()

add_executable(snr_extract
        src/main.cpp
        src/config.h src/config.cpp
        src/log.h src/log.cpp
        src/utils.h src/utils.cpp
        src/rnx_extract.h src/rnx_extract.cpp
        src/signals.h
        ${RTKLIB_SRC}/rtkcmn.c
        ${RTKLIB_SRC}/rinex.c
        ${RTKLIB_SRC}/preceph.c
)

if (UNIX OR MINGW)
  target_link_libraries(snr_extract m)
endif()
if (WIN32)
  target_link_libraries(snr_extract winmm)
endif()

target_compile_definitions(snr_extract PRIVATE
        ENAGLO ENAGAL ENAQZS ENACMP ENAIRN TRACE NFREQ=3 NEXOBS=6
)