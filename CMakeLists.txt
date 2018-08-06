cmake_minimum_required(VERSION 3.10)
project(cude_lw2)

set(CMAKE_CXX_STANDARD 11)

include_directories(inc)

add_executable(cude_lw2
        inc/lw.h
        inc/lzmp.h
        scr/main.cpp
        scr/lw.cu
        scr/lzmp.cu
        scr/max_min.cu
        scr/mat.cu
        scr/misc.cu
        scr/alg_compDistFromEx.cu
        scr/alg_initDnun.cu
        scr/dist.cu
        inc/jetbrains_ide.h inc/max_min.h inc/config.h inc/types.h inc/mat.h inc/misc.h inc/alg_compDistFromEx.h inc/dist.h inc/alg_initDnun.h)
