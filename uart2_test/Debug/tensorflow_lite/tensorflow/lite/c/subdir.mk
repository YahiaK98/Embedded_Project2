################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../tensorflow_lite/tensorflow/lite/c/common.c 

C_DEPS += \
./tensorflow_lite/tensorflow/lite/c/common.d 

OBJS += \
./tensorflow_lite/tensorflow/lite/c/common.o 


# Each subdirectory must supply rules for building sources it contributes
tensorflow_lite/tensorflow/lite/c/common.o: ../tensorflow_lite/tensorflow/lite/c/common.c
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DUSE_HAL_DRIVER -DDEBUG -DSTM32L432xx -c -I../Core/Inc -I../Drivers/STM32L4xx_HAL_Driver/Inc -I../Drivers/STM32L4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32L4xx/Include -I../Drivers/CMSIS/Include -I"D:/Media/Git/Repos/ESPrj2_TFLiteMicro_Thumper/uart2_test/tensorflow_lite" -I"D:/Media/Git/Repos/ESPrj2_TFLiteMicro_Thumper/uart2_test/tensorflow_lite/third_party/flatbuffers/include" -I"D:/Media/Git/Repos/ESPrj2_TFLiteMicro_Thumper/uart2_test/tensorflow_lite/third_party/gemmlowp" -I"D:/Media/Git/Repos/ESPrj2_TFLiteMicro_Thumper/uart2_test/tensorflow_lite/third_party/ruy" -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"tensorflow_lite/tensorflow/lite/c/common.d" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

