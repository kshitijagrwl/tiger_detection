
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[1]:

# import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json 
from collections import defaultdict
from io import StringIO
# from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.extend(["/home/kshitij/dl/tensorflow-models/research/","/home/kshitij/dl/tensorflow-models/research/object_detection"])
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


# ## Env setup

# This is needed to display the images.
# get_ipython().magic(u'matplotlib inline')
import numpy as np


# ## Object detection imports
# Here are the imports from the object detection module.

# In[ ]:

from utils import label_map_util

from utils import visualization_utils as vis_util

from matplotlib import pyplot as plt


import argparse 

def parse_args():
  parser = argparse.ArgumentParser(description='Test a SSD Inceptionv2 network')
  parser.add_argument('--path_to_ckpt', dest='PATH_TO_CKPT',
            help='Frozen model file', default=None, type=str)
  parser.add_argument('--labels', dest='PATH_TO_LABELS',
            help='Label proto file',
            default=None, type=str)

  parser.add_argument('--test_img_path', dest='TEST_IMAGE_PATHS',
            help='Path to test image directory',
            default=None, type=str)

  parser.add_argument('--output_path', dest='output_path', 
            help='competition mode', type=str)
  
  parser.add_argument('--num_dets', dest='max_per_image',
            help='max number of detections per image',
            default=100, type=int)
  parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152, mobile',
                      default='res50', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'model_3/' + 'frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('tiger_label_map.pbtxt')
NUM_CLASSES = 2



## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.

import glob
  
# Val
# im_nums = np.loadtxt('/archive/tiger_in_wild/ImageSets/Main/val.txt',dtype=np.int)
# TEST_IMAGE_PATHS = [str('/archive/tiger_in_wild/JPEGImages/{:04d}.jpg'.format(i)) for i in im_nums]

# Test 
TEST_IMAGE_PATHS = glob.glob('/archive/tiger_in_wild/det_test/*jpg')
TEST_IMAGE_PATHS.sort()

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


output_path = 'output_imgs/model_3/test/'
output_json = []
base_json = {}


THRESH = 0.4
for image_path in TEST_IMAGE_PATHS:
  image = Image.open(image_path)
  (im_width, im_height) = image.size 
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
  image_id = int(image_path.strip('.jpg').split('/')[-1])
  print ("Image {}".format(image_id))

  for i in xrange(len(output_dict['detection_boxes'])):
    
    if output_dict['detection_scores'][i]> THRESH:
      base_json = {}
      base_json['image_id'] = image_id
      base_json['category_id'] = 1

      base_json['bbox'] = [output_dict['detection_boxes'][i][1]*im_width,output_dict['detection_boxes'][i][0]*im_height,
                          (output_dict['detection_boxes'][i][3]-output_dict['detection_boxes'][i][1])*im_width,(output_dict['detection_boxes'][i][2]-output_dict['detection_boxes'][i][0])*im_height]

      # base_json['bbox'] = output_dict['detection_boxes'][i].tolist()
      base_json['score'] = float(np.around(output_dict['detection_scores'][i],4))
      # print(img_json['bbox'])

      output_json.extend([base_json])
  # 

  # write outputs 
  Image.fromarray(image_np).save(output_path+str(base_json['image_id']) + ".jpg")

# print(output_json)
fp = open(output_path+"detections_50.json","w")
json.dump(output_json,fp)







