# detection-pipeline-tpu
A simple framework wrapping google's work on developing custom detector on any objects, ready to deploy on TPU. 

# face_detection_TPU
 ## Steps to follow 
(1). Create a virtual environment
```
$ conda create -n tensorflow_lite python=3.9
```
(2). activate the virtual environment.

(3). Setup the coral environment, using the [setting up coral](https://coral.ai/docs/accelerator/get-started/)

(4). install all the dependencies to use face detection
```
$ bash install_requirements.sh
```
(5). Run the script to face detection
```
$ python3 detect.py
```
## Pretrained Models:

This repository consists of different versions of the efficient Det models. . There are several model sizes you can choose from:

|| Model architecture | Size(MB)* | Latency(ms)** | Average Precision*** |
|-|--------------------|-----------|---------------|----------------------|
|| EfficientDet-Lite0 | 5.7       | 37.4            | 30.4%               |
|| EfficientDet-Lite1 | 7.6       | 56.3            | 34.3%               |
|| EfficientDet-Lite2 | 10.2      | 104.6           | 36.0%               |
|| EfficientDet-Lite3 | 14.4      | 107.6           | 39.4%               |


## Models Finetuned:

The models were different epoch values and batch size. The hyperparameters used for finetuning are mentioned in the table below:

|| Model architecture |Batch Size| Epochs        | Model Link|
|-|-------------------|----------|---------------|----------------------|
|| EfficientDet-Lite0 | 64       | 1000          |[model1](all_models/efficientdet-lite-face_1000e_64b_edgetpu.tflite)
|| EfficientDet-Lite1 | 32       | 400           |[model2](all_models/efficientdet-lite1-face_400e_32b_edgetpu.tflite)
|| EfficientDet-Lite2 | 16       | 300           |[model3](all_models/efficientdet-lite2-face_300e_16b_edgetpu.tflite)
|| EfficientDet-Lite3 | 8        | 200           |[model4]()

The pre trained models with the above parameters are saved in the all models and can be accessed by the following command in  *Line 36* of **detect.py** script:

```
$ default_model = 'efficient_lite_face_edgetpu.tflite'
```
the choice of the models are given below:

```
efficientdet-lite0_small_ax_face_300e_64b_edgetpu.tflite
efficientdet-lite0_small_ax_face_220e_64b_edgetpu.tflite
efficientdet-lite0_small_ax_face_150e_64b_edgetpu.tflite
```
The above pretrained models can be used by adding the path of the saved model in Line 36* of **detect.py** script:
```
$ default_model = '...'
```
## Curating the Dataset:
The custom dataset used for the model training is extracted using a zip  file called *dataset.zip*. 

### Structure of the Dataset:
```
├──  dataset  
    └── train
        └──images
        └──annotations  
    └── validation
        └──images
        └──annotations   
    └── test
        └──images
        └──annotations

```
The annotations consist of bounding box co-ordinates of the objects in .xml format which is saved in *dataset/train/annotations*

Labels of the images for training are saved in **face_labels.txt** and can have multiple labels. 