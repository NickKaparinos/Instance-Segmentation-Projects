# Instance-Segmentation-Projects
Pedestrian and Balloon Instance Segmentation projects using Pytorch and Mask R-CNN.

## Instance Segmentation
Instance segmentation is a challenging computer vision task that requires the prediction of object instances and their per-pixel segmentation mask. This makes it a hybrid of semantic segmentation and object detection.

## Mask R-CNN
Deep Learning methodologies are able to do end-to-end object detection using Convolutional Neural Networks. The algorithms designed to do object detection are based on two approaches - one-stage object detection and two-stage object detection. One-stage detectors have high inference speeds and two-stage detectors have high localization and recognition accuracy. The two stages of a two-stage detector can be divided by a RoI (Region of Interest) Pooling layer. One of the prominent two-stage object detectors is Faster R-CNN. It has the first stage called RPN, a Region Proposal Network to predict candidate bounding boxes. In the second stage, features are by RoI pooling operation from each candidate box for the following classification and boundingbox regression tasks. Mask R-CNN is an extention of Faster R-CNN for instance segmentation. 
[[Bibliometric Analysis of One-stage and Two-stage Object Detection]](https://digitalcommons.unl.edu/libphilprac/4910/)

## Pedestrian Instance Segmentation

### Training and Evaluation Learning Curve
<p float="left">
  <img src="https://github.com/NickKaparinos/Instance-Segmentation-Projects/blob/master/images/Pedestrian_learning_curve.jpg" width="49.7%" /> 
  <img src="https://github.com/NickKaparinos/Instance-Segmentation-Projects/blob/master/images/Pedestrian_validation_metrics.jpg" width="49.7%" />
</p>

### Inference examples and comparison to Ground Truth

||||
|:---:|:---:|:---:|
| Image &nbsp | Predictions | Ground Truth |





|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Image&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Predictions&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ground Truth&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
|:---------------------------:|:---:|:---:|


<p float="left">
  <img src="https://github.com/NickKaparinos/Instance-Segmentation-Projects/blob/master/images/Pedestrian_image_1.jpg" width="33.0%" /> 
  <img src="https://github.com/NickKaparinos/Instance-Segmentation-Projects/blob/master/images/Pedestrian_image_1_segmentation.jpg" width="33.0%" />
  <img src="https://github.com/NickKaparinos/Instance-Segmentation-Projects/blob/master/images/Pedestrian_image_1_ground_truth_segmentation.jpg" width="33.0%" />
</p>




## Balloon Instance Segmentation
### Training and Evaluation Learning Curve
### Inference examples and comparison to Ground Truth
