# !/usr/bin/env python
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Location of stereo datasets.
DATASETS_LOCATION="/tmp/datasets/"
MODEL_PATH="/tmp/hitnet/models"


# max_disparity = 160
# MODEL_NAME="middlebury_d160.pb"
# max_disparity = 288
# MODEL_NAME="middlebury_d288.pb"
# max_disparity = 400
MODEL_NAME="middlebury_d400.pb"
# DATA_PATTERN="$DATASETS_LOCATION/middlebury/trainingH/"
DATA_PATTERN="/Users/nihalkurki/Desktop/techlab/hitnet/"
# DATA_PATTERN="$DATASETS_LOCATION/middlebury/testH/"
LEFT_PATTERN="left.png"
RIGHT_PATTERN="right.png"
GT_LEFT_PATTERN="disparity_pfm.pfm"
# /Users/nihalkurki/Desktop/techlab/hitnet/middle_left.png


#!/bin/bash
set -e
set -x

python3 -m venv pyenv_tf2
source pyenv_tf2/bin/activate

python3 -m pip install -r requirements.txt
python3 -m pip install opencv-python

mkdir -p $MODEL_PATH
# wget -P $MODEL_PATH -N https://storage.googleapis.com/tensorflow-graphics/models/hitnet/default_models/$MODEL_NAME

python -m predict \
  --data_pattern=$DATA_PATTERN \
  --model_path="/Users/nihalkurki/Desktop/techlab/hitnet/middlebury_d400.pb" \
  --png_disparity_factor=128 \
  --iml_pattern=$LEFT_PATTERN \
  --imr_pattern=$RIGHT_PATTERN \
  --gtl_pattern=$GT_LEFT_PATTERN \
  --input_channels=3 \
  --predict_right=false \
  --save_png=true \
  --save_pfm=false \
  --evaluate=true \
  --max_test_number=10000 \

# factor before was 128
