/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <micro_model_settings.h>
#include "audio_provider.h"
#include "no_1000ms_sample_data.h"
#include "yes_1000ms_sample_data.h"


namespace {
int16_t g_dummy_audio_data[kMaxAudioSampleSize];
int32_t g_latest_audio_timestamp = 0;
}  // namespace


TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples, int16_t* samples) {
  const int yes_start = (0 * kAudioSampleFrequency) / 1000;
  const int yes_end = (1000 * kAudioSampleFrequency) / 1000;
  const int start_sample = (start_ms * kAudioSampleFrequency) / 1000;
  for (int i = 0; i < kMaxAudioSampleSize; ++i) {
    const int sample_index = (start_sample + i);
    int16_t sample;
    if ((sample_index >= yes_start) && (sample_index < yes_end)) {
      sample = samples[sample_index - yes_start];
    } else {
      sample = 0;
    }
    g_dummy_audio_data[i] = sample;
  }
  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_dummy_audio_data;
  return kTfLiteOk;
}

int32_t LatestAudioTimestamp() {
  g_latest_audio_timestamp += 100;
  return g_latest_audio_timestamp;
}
