################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
S_SRCS += \
../Core/Startup/startup_stm32l432kcux.s 

S_DEPS += \
./Core/Startup/startup_stm32l432kcux.d 

OBJS += \
./Core/Startup/startup_stm32l432kcux.o 


# Each subdirectory must supply rules for building sources it contributes
Core/Startup/startup_stm32l432kcux.o: ../Core/Startup/startup_stm32l432kcux.s
	arm-none-eabi-gcc -mcpu=cortex-m4 -g3 -c -I"D:/Media/Git/Repos/ESPrj2_TFLiteMicro_Thumper/uart2_test/tensorflow_lite" -I"D:/Media/Git/Repos/ESPrj2_TFLiteMicro_Thumper/uart2_test/tensorflow_lite/third_party/flatbuffers/include" -I"D:/Media/Git/Repos/ESPrj2_TFLiteMicro_Thumper/uart2_test/tensorflow_lite/third_party/gemmlowp" -I"D:/Media/Git/Repos/ESPrj2_TFLiteMicro_Thumper/uart2_test/tensorflow_lite/third_party/ruy" -x assembler-with-cpp -MMD -MP -MF"Core/Startup/startup_stm32l432kcux.d" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@" "$<"

