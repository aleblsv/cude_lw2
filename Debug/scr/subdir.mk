################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../scr/main.cpp 

CU_SRCS += \
../scr/alg_compDistFromEx.cu \
../scr/alg_initDnun.cu \
../scr/dist.cu \
../scr/lw.cu \
../scr/lzmp.cu \
../scr/mat.cu \
../scr/max_min.cu \
../scr/misc.cu 

CU_DEPS += \
./scr/alg_compDistFromEx.d \
./scr/alg_initDnun.d \
./scr/dist.d \
./scr/lw.d \
./scr/lzmp.d \
./scr/mat.d \
./scr/max_min.d \
./scr/misc.d 

OBJS += \
./scr/alg_compDistFromEx.o \
./scr/alg_initDnun.o \
./scr/dist.o \
./scr/lw.o \
./scr/lzmp.o \
./scr/main.o \
./scr/mat.o \
./scr/max_min.o \
./scr/misc.o 

CPP_DEPS += \
./scr/main.d 


# Each subdirectory must supply rules for building sources it contributes
scr/alg_compDistFromEx.o: /home/alex/cuda-workspace/lw/scr/alg_compDistFromEx.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-8.0/samples/6_Advanced -I"/home/alex/cuda-workspace/lw/inc" -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "scr" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-8.0/samples/6_Advanced -I"/home/alex/cuda-workspace/lw/inc" -G -g -O0 --compile --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

scr/alg_initDnun.o: /home/alex/cuda-workspace/lw/scr/alg_initDnun.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-8.0/samples/6_Advanced -I"/home/alex/cuda-workspace/lw/inc" -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "scr" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-8.0/samples/6_Advanced -I"/home/alex/cuda-workspace/lw/inc" -G -g -O0 --compile --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

scr/dist.o: /home/alex/cuda-workspace/lw/scr/dist.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-8.0/samples/6_Advanced -I"/home/alex/cuda-workspace/lw/inc" -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "scr" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-8.0/samples/6_Advanced -I"/home/alex/cuda-workspace/lw/inc" -G -g -O0 --compile --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

scr/lw.o: /home/alex/cuda-workspace/lw/scr/lw.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-8.0/samples/6_Advanced -I"/home/alex/cuda-workspace/lw/inc" -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "scr" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-8.0/samples/6_Advanced -I"/home/alex/cuda-workspace/lw/inc" -G -g -O0 --compile --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

scr/lzmp.o: /home/alex/cuda-workspace/lw/scr/lzmp.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-8.0/samples/6_Advanced -I"/home/alex/cuda-workspace/lw/inc" -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "scr" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-8.0/samples/6_Advanced -I"/home/alex/cuda-workspace/lw/inc" -G -g -O0 --compile --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

scr/main.o: /home/alex/cuda-workspace/lw/scr/main.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-8.0/samples/6_Advanced -I"/home/alex/cuda-workspace/lw/inc" -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "scr" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-8.0/samples/6_Advanced -I"/home/alex/cuda-workspace/lw/inc" -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

scr/mat.o: /home/alex/cuda-workspace/lw/scr/mat.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-8.0/samples/6_Advanced -I"/home/alex/cuda-workspace/lw/inc" -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "scr" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-8.0/samples/6_Advanced -I"/home/alex/cuda-workspace/lw/inc" -G -g -O0 --compile --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

scr/max_min.o: /home/alex/cuda-workspace/lw/scr/max_min.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-8.0/samples/6_Advanced -I"/home/alex/cuda-workspace/lw/inc" -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "scr" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-8.0/samples/6_Advanced -I"/home/alex/cuda-workspace/lw/inc" -G -g -O0 --compile --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

scr/misc.o: /home/alex/cuda-workspace/lw/scr/misc.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-8.0/samples/6_Advanced -I"/home/alex/cuda-workspace/lw/inc" -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "scr" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda-8.0/samples/6_Advanced -I"/home/alex/cuda-workspace/lw/inc" -G -g -O0 --compile --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


