# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo which runs object classification on camera frames.

Run default object detection:
python3 classify.py

Choose different camera and input encoding
python3 classify.py --videosrc /dev/video1 --videofmt jpeg
"""

import argparse # 定義輸入參數
import gstreamer # 一个多媒体框架，它可以允许你轻易地创建、编辑与播放多媒体文
import os
import time

from common import avg_fps_counter, SVG
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
from pycoral.adapters.common import input_size
from pycoral.adapters.classify import get_classes # Gets results from a classification model as a list of ordered classes.

def generate_svg(size, text_lines):
    svg = SVG(size)
    for y, line in enumerate(text_lines, start=1):
      svg.add_text(10, y * 20, line, 20)
    return svg.finish()

def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_v2_1.0_224_quant_edgetpu.tflite'
    default_labels = 'imagenet_labels.txt'
    parser = argparse.ArgumentParser()
    # 利用 add_argument 可以指名讓我們的程式接受哪些命令列參數
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
                        # 至多顯示幾種辨識出類別，預設為前 3 明顯
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
                        # 最低辨識準確度顯示，至少 10 % matched
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    parser.add_argument('--videosrc', help='Which video source to use. ',
                        default='/dev/video0')
                        # 執行時是否不顯示攝像頭拍攝鏡頭，預設為要顯示
    parser.add_argument('--headless', help='Run without displaying the video.',
                        default=False, type=bool)
    parser.add_argument('--videofmt', help='Input video format.',
                        default='raw',
                        choices=['raw', 'h264', 'jpeg'])
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    # make_interpreter: Creates a new tf.lite.Interpreter instance using the given model.
    interpreter = make_interpreter(args.model)
    # interpreter.allocate_tensors():
    # Since TensorFlow Lite pre-plans tensor allocations to optimize inference, 
    # the user needs to call allocate_tensors() before any inference.
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    # Average fps over last 30 frames.
    fps_counter = avg_fps_counter(30)

    def user_callback(input_tensor, src_size, inference_box):
      nonlocal fps_counter # 透過關鍵字 nonlocal 說明函數要修改上層函數定義的局部變數
      start_time = time.monotonic() # count time
      run_inference(interpreter, input_tensor)
      # get_classes: Gets results from a classification model as a list of ordered classes.
      results = get_classes(interpreter, args.top_k, args.threshold)
      end_time = time.monotonic()
      text_lines = [
          ' ',
          'Inference: {:.2f} ms'.format((end_time - start_time) * 1000),
          'FPS: {} fps'.format(round(next(fps_counter))),
      ]
      for result in results:
          text_lines.append('score={:.2f}: {}'.format(result.score, labels.get(result.id, result.id)))
      print(' '.join(text_lines))
      return generate_svg(src_size, text_lines)

    result = gstreamer.run_pipeline(user_callback,
                                    src_size=(640, 480), # camera Image resolution
                                    appsink_size=inference_size, # mointor size
                                    videosrc=args.videosrc,
                                    videofmt=args.videofmt,
                                    headless=args.headless)
    gstreamer.run_pipeline

if __name__ == '__main__':
    main()
