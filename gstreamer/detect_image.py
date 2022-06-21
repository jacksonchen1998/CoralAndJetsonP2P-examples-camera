# Lint as: python3
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
r"""Example using PyCoral to detect objects in a given image.

To run this code, you must attach an Edge TPU attached to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.

Example usage:
```
bash examples/install_requirements.sh detect_image.py

python3 examples/detect_image.py \
  --model test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels test_data/coco_labels.txt \
  --input test_data/grace_hopper.bmp \
  --output ${HOME}/grace_hopper_processed.bmp
```
"""

import argparse
import time
import cv2

from PIL import Image
from PIL import ImageDraw

from common import avg_fps_counter
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='red')


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', help='File path of .tflite file',
                      default='test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite')
                      # Model from package called pycoral
  parser.add_argument('-i', '--input',
                      help='File path of image to process',
                      default="coral_src1.jpg")
  parser.add_argument('-l', '--labels', help='File path of labels file',
                      default='test_data/coco_labels.txt')
  parser.add_argument('-t', '--threshold', type=float, default=0.4,
                      help='Score threshold for detected objects')
  parser.add_argument('-o', '--output',
                      help='File path for the result image with annotations',
                      default="output.jpg")
  parser.add_argument('-c', '--count', type=int, default=1,
                      help='Number of times to run inference')
  args = parser.parse_args()

  file_fps = open("p2p_detect_fps.txt", "w")
  file_detail = open("p2p_detect.txt", "w")

  labels = read_label_file(args.labels) if args.labels else {}
  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  fps_counter = avg_fps_counter(30)

  cam_port = 1
  cam = cv2.VideoCapture(cam_port)

  while True:
    result, image = cam.read()

    if result:
      cv2.imwrite("coral_src1.jpg", "w")
    else:
      print("No image")

    image = Image.open("coral_src1.jpg")
    _, scale = common.set_resized_input(
      interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

    #print('----INFERENCE TIME----')
    #print('Note: The first inference is slow because it includes',
    #'loading the model into Edge TPU memory.')
    for _ in range(args.count):
      start = time.perf_counter()
      interpreter.invoke()
      inference_time = time.perf_counter() - start
      objs = detect.get_objects(interpreter, args.threshold, scale)
      #file_fps.write(str('%.2f' % (inference_time * 1000)))
      file_fps.write(str(round(next(fps_counter)))
      file_fps.flush()

    #print('-------RESULTS--------')
    if not objs:
      print('No objects detected')

    for obj in objs:
      file_detail.write(str(labels.get(obj.id, obj.id)) + "\n")
      file_detail.write(str(obj.id) + "\n")
      file_detail.write(str(obj.score) + "\n")
      #print('  bbox:  ', obj.bbox)
      file_detail.flush()

    if args.output:
      image = image.convert('RGB')
      draw_objects(ImageDraw.Draw(image), objs, labels)
      image.save(args.output)
      image.show()

if __name__ == '__main__':
  main()
