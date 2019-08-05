# Tiger Detection 

The object detection model is based on SSD-InceptionV2 model, pretrained on the COCO Detection task. 

The model definition is from - [Tensorflor Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

This repository contains the:
1. Model definition config
2. Class label proto
3. Object Prediction code

## Running the Code

sys.path.append("path-to-tf-models-object-detection-folder") 
Update TEST_IMAGE_PATHS
Update output_path 

Run using : 
``` python object_prediction.py ```

The file output will be stored in 
``` detections_50.json ```



