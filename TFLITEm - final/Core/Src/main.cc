/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2020 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "main_functions.h"

#include "model.h"
#include "yes_micro_features_data.h"
#include "micro_error_reporter.h"
#include "micro_interpreter.h"
#include "micro_mutable_op_resolver.h"
//#include "micro_test.h"
#include "schema_generated.h"
#include "version.h"
#include "feature_provider.h"
#include "micro_model_settings.h"
#include "audio_provider.h"
#include "recognize_commands.h"

using namespace std;
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;

CRC_HandleTypeDef hcrc;

TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim16;

UART_HandleTypeDef huart1;
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
int sampleCompleted;
char message[32];
bool flag = false;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_TIM16_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_CRC_Init(void);
static void MX_ADC1_Init(void);
static void MX_TIM1_Init(void);
static void MX_USART1_UART_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
char buff[2];
int fwd_delay = 1500;//3100; // value was set to 4800ms but due to battery power error we found that 3100ms made it move 2m


void forward(void)
{
	// Forward
		buff[1] = 0x40; // Set speed to be 50% of max (1.5 km/h)
		buff[0] = 0xC2;
		HAL_UART_Transmit(&huart1, (uint8_t *) buff, sizeof(buff), 200);
		buff[0] = 0xCA;
		HAL_UART_Transmit(&huart1, (uint8_t *) buff, sizeof(buff), 200);

		//HAL_Delay(fwd_delay); // Delay to move forward for a set time

}

void left(void)
{
	// Turn Left
		//buff[1] = 0x06; //set slower speed to 6 (calculated above)
		//buff[0] = 0xC2;
		buff[1] = 0x40; //set slower speed to 6 (calculated above)
		buff[0] = 0xC1;
		HAL_UART_Transmit(&huart1, (uint8_t *) buff, sizeof(buff), 200);

		buff[1] = 0x40;
		buff[0] = 0xCA;
		HAL_UART_Transmit(&huart1, (uint8_t *) buff, sizeof(buff), 200);

		HAL_Delay(530); // since turning speed of 5.6 was calculated for 1500ms -> we changed delay to 1400 to account for the fact that we're using 6 instead

}

void right(void)
{
			// Turn Right
		buff[1] = 0x40;
		buff[0] = 0xC2;
		HAL_UART_Transmit(&huart1, (uint8_t *) buff, sizeof(buff), 200);

		//buff[1] = 0x06;
		//buff[0] = 0xCA;
		buff[1] = 0x40;
		buff[0] = 0xC9;
		HAL_UART_Transmit(&huart1, (uint8_t *) buff, sizeof(buff), 200);

		HAL_Delay(550); // we found that the left motor was weaker -> the same delay made it turn less degrees -> increase delay to account for weaker motor


}

void brake(void)
{
	// Brake
		buff[1] = 0x00;
		buff[0] = 0xC0;
		HAL_UART_Transmit(&huart1, (uint8_t *) buff, sizeof(buff), 200);
		buff[0] = 0xC8;
		HAL_UART_Transmit(&huart1, (uint8_t *) buff, sizeof(buff), 200);

		//HAL_Delay(500);

}
void moveCar(int state){
	if(state==0)
		brake();
	else if(state==1)
		forward();
	else if(state==2)
		left();
	else
		right();
}

const int no_of_samples = 64000;
//int16_t samples[no_of_samples];


int readSamples()
{

	//char debug [10];
//	int a =0;
	//int read;
	int bit, toggle, prescalerCounter=0;
	int debouncerPositiveCounter =0;
	int debouncerNegativeCounter=0;
	int stateTriggered= 0;
	//float convf;
	//float volt;
	__HAL_TIM_SET_COUNTER(&htim1, 0);
	toggle=0;
	while(sampleCompleted<no_of_samples)
	{
		while (__HAL_TIM_GET_COUNTER(&htim1) < 10);

		prescalerCounter +=1;
		if(prescalerCounter>1)
		{
			__HAL_TIM_SET_COUNTER(&htim1, 0);

			if(HAL_GPIO_ReadPin(GPIOB,GPIO_PIN_5))
				bit =0;
			else
				bit =1;

			if(bit){
				debouncerPositiveCounter +=1;
				if(debouncerPositiveCounter>10){
					if(stateTriggered==0)
						toggle+=1;
					stateTriggered=1;
				}

				debouncerNegativeCounter = 0;
			}else{
				debouncerNegativeCounter +=1;
			}

			if(debouncerNegativeCounter>600){
				stateTriggered =0;
				debouncerPositiveCounter=0;
				debouncerNegativeCounter=0;
			}



			prescalerCounter=0;
		}

		sampleCompleted++;

	}


	sprintf(message,"NOISY  = %d TOGGLE = %d \r\n", bit,toggle);
	HAL_UART_Transmit(&huart2,(uint8_t *)&message, 25, 100);
	sampleCompleted = 0;
	if(toggle<4)
		return toggle;
	else return 0;
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */
	FeatureProvider* feature_provider = nullptr;
	RecognizeCommands* recognizer = nullptr;
	int8_t feature_buffer[kFeatureElementCount];
	uint32_t timestamp;
	char buf[50];
	int buf_len = 0;
	//int16_t samples[1];
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_TIM16_Init();
  MX_USART2_UART_Init();
  MX_CRC_Init();
  MX_ADC1_Init();
  MX_TIM1_Init();
  MX_USART1_UART_Init();
  /* USER CODE BEGIN 2 */
  //setup();

  // Set up logging.
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = ::tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
	  TF_LITE_REPORT_ERROR(&micro_error_reporter,
					   "Model provided is schema version %d not equal "
					   "to supported version %d.\n",
					   model->version(), TFLITE_SCHEMA_VERSION);
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  tflite::MicroMutableOpResolver<4> micro_op_resolver;
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();

  // Create an area of memory to use for input, output, and intermediate arrays.
  const int tensor_arena_size = 10 * 1024;
  uint8_t tensor_arena[tensor_arena_size];

  // Build an interpreter to run the model with.
  tflite::MicroInterpreter interpreter(model, micro_op_resolver, tensor_arena,
									 tensor_arena_size,
									 &micro_error_reporter);
  interpreter.AllocateTensors();

  // Get information about the memory area to use for the model's input.
  TfLiteTensor* input = interpreter.input(0);
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                   feature_buffer);
  feature_provider = &static_feature_provider;

  static RecognizeCommands static_recognizer(error_reporter);
  recognizer = &static_recognizer;

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */

   //float convF;
   //uint32_t converted;
   HAL_TIM_Base_Start (&htim1);
   HAL_TIM_Base_Start(&htim16);
   //volatile int bit;
   __HAL_TIM_ENABLE_IT(&htim1,TIM_IT_CC1);
   int32_t previous_time = 0;
   HAL_UART_Transmit(&huart2,(uint8_t *)"STRT!\r\n", 7, 100);

   const int8_t* go_features_data = g_go_micro_f2e59fea_nohash_1_data;
   const int8_t* stop_features_data = g_stop_micro_f2e59fea_nohash_1_data;
   const int8_t* right_features_data = g_right_micro_f2e59fea_nohash_1_data;
   const int8_t* left_features_data = g_left_micro_f2e59fea_nohash_1_data;

   int counter = 0;
   int carState =0;
   int prevCarState =0;
   int firstLoop = 1;
   int detectedState=0;
   int changeOfState =1;
   // 0 Stop
   // 1 Go
   // 2 Left
   // 3 Right

   while(1)
   {
	   if(firstLoop==0)
	   {
			detectedState = readSamples();
			prevCarState = carState;

			if(detectedState ==1){
				changeOfState =1;
				if(carState == 0 || carState == 1)
					carState = (carState+1)%2;
			} else if (detectedState != 0){



				carState = detectedState;
			}

			changeOfState = (carState!=prevCarState);
		}
	   else
	   {
			firstLoop=false;
	   }

	   	int how_many_new_slices = 0;

		// Copy a spectrogram created from a .wav audio file of someone saying respective "Command",
		// into the memory area used for the input.
		//const int8_t* yes_features_data = g_go_micro_f2e59fea_nohash_1_data;

	   	for (size_t i = 0; i < input->bytes; ++i)
	   	{
			if(carState == 0)
				input->data.int8[i] = stop_features_data[i];
			else if(carState == 1)
				input->data.int8[i] = go_features_data[i];
			else if(carState == 2)
				input->data.int8[i] = left_features_data[i];
			else if(carState == 3)
				input->data.int8[i] = right_features_data[i];
	   	}

	   	// Run the model on this input and make sure it succeeds.
	   	if(changeOfState)
	   	{
			changeOfState = 0;
			timestamp = htim16.Instance->CNT;
			TfLiteStatus invoke_status = interpreter.Invoke();
			if (invoke_status != kTfLiteOk)
			{
			  TF_LITE_REPORT_ERROR(&micro_error_reporter, "Invoke failed\n");
			}

			// Get the output from the model, and make sure it's the expected size and
			// type.
			TfLiteTensor* output = interpreter.output(0);

			 buf_len = sprintf(buf,
			                      "Inference took: %lu\r\n",
			                      htim16.Instance->CNT - timestamp);
			 HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);

			bool is_new_command = false;
			int watchdogCounter = 0;
			const char* found_command = nullptr;
			while(!is_new_command && watchdogCounter<20)
			{
				watchdogCounter++;
				const int32_t current_time = LatestAudioTimestamp();

				uint8_t score = 0;

				TfLiteStatus process_status = recognizer->ProcessLatestResults(
				output, current_time, &found_command, &score, &is_new_command);
				if (process_status != kTfLiteOk)
				{
					error_reporter->Report("RecognizeCommands::ProcessLatestResults() failed");
					return -1;
				}

				HAL_UART_Transmit(&huart2, (uint8_t *)"\n\r Heard: ", 10, 100);
				HAL_UART_Transmit(&huart2, (uint8_t *)found_command, 2, 100);
				HAL_UART_Transmit(&huart2, (uint8_t *)"  State:  ", 10, 100);
				char buf2[100]="";

				int len = sprintf(buf2, "%d ",carState);
				HAL_UART_Transmit(&huart2, (uint8_t *)buf2, len, 100);
			}
			buf_len = sprintf(buf,
						          "Command Recognition took: %lu\r\n",
						          htim16.Instance->CNT - timestamp);
						 HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
			int tempState = 0;

			if(!strcmp(found_command,"stop"))
			  tempState =0;
			else if(!strcmp(found_command,"go"))
			  tempState =1;
			else if(!strcmp(found_command,"left") )
			  tempState =2;
			else if(!strcmp(found_command,"right"))
			  tempState =3;

			moveCar(tempState);
			if(carState ==2 || carState==3)
			{
				prevCarState = prevCarState==2||prevCarState==3? 0 : prevCarState;
				carState =  prevCarState;

				moveCar(carState);
			}
	   	}
   	}

  //while (1)
  //{
	//loop();
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  //}
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

  /** Configure LSE Drive Capability
  */
  HAL_PWR_EnableBkUpAccess();
  __HAL_RCC_LSEDRIVE_CONFIG(RCC_LSEDRIVE_LOW);
  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_LSE|RCC_OSCILLATORTYPE_MSI;
  RCC_OscInitStruct.LSEState = RCC_LSE_ON;
  RCC_OscInitStruct.MSIState = RCC_MSI_ON;
  RCC_OscInitStruct.MSICalibrationValue = 0;
  RCC_OscInitStruct.MSIClockRange = RCC_MSIRANGE_6;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_MSI;
  RCC_OscInitStruct.PLL.PLLM = 1;
  RCC_OscInitStruct.PLL.PLLN = 40;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV7;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
  PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_USART1|RCC_PERIPHCLK_USART2
                              |RCC_PERIPHCLK_ADC;
  PeriphClkInit.Usart1ClockSelection = RCC_USART1CLKSOURCE_PCLK2;
  PeriphClkInit.Usart2ClockSelection = RCC_USART2CLKSOURCE_PCLK1;
  PeriphClkInit.AdcClockSelection = RCC_ADCCLKSOURCE_SYSCLK;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
  {
    Error_Handler();
  }
  /** Configure the main internal regulator output voltage
  */
  if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1) != HAL_OK)
  {
    Error_Handler();
  }
  /** Enable MSI Auto calibration
  */
  HAL_RCCEx_EnableMSIPLLMode();
}

/**
  * @brief ADC1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC1_Init(void)
{

  /* USER CODE BEGIN ADC1_Init 0 */

  /* USER CODE END ADC1_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */
  /** Common config
  */
  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_ASYNC_DIV1;
  hadc1.Init.Resolution = ADC_RESOLUTION_12B;
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.ScanConvMode = ADC_SCAN_DISABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  hadc1.Init.LowPowerAutoWait = DISABLE;
  hadc1.Init.ContinuousConvMode = DISABLE;
  hadc1.Init.NbrOfConversion = 1;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc1.Init.DMAContinuousRequests = DISABLE;
  hadc1.Init.Overrun = ADC_OVR_DATA_PRESERVED;
  hadc1.Init.OversamplingMode = DISABLE;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }
  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_12;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime = ADC_SAMPLETIME_2CYCLES_5;
  sConfig.SingleDiff = ADC_SINGLE_ENDED;
  sConfig.OffsetNumber = ADC_OFFSET_NONE;
  sConfig.Offset = 0;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */

}

/**
  * @brief CRC Initialization Function
  * @param None
  * @retval None
  */
static void MX_CRC_Init(void)
{

  /* USER CODE BEGIN CRC_Init 0 */

  /* USER CODE END CRC_Init 0 */

  /* USER CODE BEGIN CRC_Init 1 */

  /* USER CODE END CRC_Init 1 */
  hcrc.Instance = CRC;
  hcrc.Init.DefaultPolynomialUse = DEFAULT_POLYNOMIAL_ENABLE;
  hcrc.Init.DefaultInitValueUse = DEFAULT_INIT_VALUE_ENABLE;
  hcrc.Init.InputDataInversionMode = CRC_INPUTDATA_INVERSION_NONE;
  hcrc.Init.OutputDataInversionMode = CRC_OUTPUTDATA_INVERSION_DISABLE;
  hcrc.InputDataFormat = CRC_INPUTDATA_FORMAT_BYTES;
  if (HAL_CRC_Init(&hcrc) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN CRC_Init 2 */

  /* USER CODE END CRC_Init 2 */

}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 499;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 19;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim1, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterOutputTrigger2 = TIM_TRGO2_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */

}

/**
  * @brief TIM16 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM16_Init(void)
{

  /* USER CODE BEGIN TIM16_Init 0 */

  /* USER CODE END TIM16_Init 0 */

  /* USER CODE BEGIN TIM16_Init 1 */

  /* USER CODE END TIM16_Init 1 */
  htim16.Instance = TIM16;
  htim16.Init.Prescaler = 79;
  htim16.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim16.Init.Period = 65535;
  htim16.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim16.Init.RepetitionCounter = 0;
  htim16.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim16) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM16_Init 2 */

  /* USER CODE END TIM16_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 19200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  huart2.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart2.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin : PB5 */
  GPIO_InitStruct.Pin = GPIO_PIN_5;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */

  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     tex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
