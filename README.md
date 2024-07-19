# Learning Cruxes to Push
This repo is the official implementation of [Learning Cruxes to Push for Object Detection in
Low-Quality Images](). It is based on [mmdetection](https://github.com/open-mmlab/mmdetection/tree/2.x) (v2.28.2).
## Introduction
In a low-quality image, false detection(negatives or positives) is more likely to occur in local regions. In this paper, we propose a simple yet effective strategy with
two learners to solve false detection. We devise the crux
learner to generate cruxes that have great impacts on detection
performance. The catch-up leaner with a simple residual transfer
mechanism maps the feature distributions of crux regions to those
favouring a deep detector. These two learners can be plugged into
any CNN-based feature extraction networks, and yield high detection accuracy on various
degraded scenarios. Extensive experiments on several public
datasets demonstrate that our method achieves more promising
results than state-of-the-art detection approaches.
![fig1](./assets/fig1.jpg)
## Datasets and Learning Strategy
Utilizing a low-quality set and a detection-favored set, we develop a cooperative learning scheme that alternatively
updates the crux and catch-up learner.

As shown in the top part of Fig. 3 (a), we keep the crux
learner fixed and update the catch-up learner using low-quality images and detection-favored images. The
catch-up learner is updated to learn to bridge the gap between the
two feature distributions.

As is shown in the bottom part of Fig.3 (a), we fix the catch-up learner and update the crux-learner, the deep stage, RPN, Neck, and Head. This update step is only updated on low-quality images with semantic labels. After the cooperative learning process, the crux learner can
identify cruxes on a low-quality image according to detection performance. 

We use URPC2020, Foggy-Cityscapes and Rainy-Cityscapes as low-quality sets to evaluate our method on underwater, foggy and rainy scenes separately. 

We use Cityscapes as the detection-favored image set for both rainy and foggy scenes. For the underwater scene, we use DFUI as the detection-favored image set.

- The URPC2020 dataset can be downloaded from [here]().
- The DFUI dataset can be downloaded from [here]().
- The synthesized Rainy-Cityscapes images can be downloaded from [here]().
- The annotations of both Rainy and Foggy Cityscaptes can be downloaded from [here]().


![fig3](./assets/fig3.jpg)
## Models and Results
### Foggy Scene (Foggy-Cityscapes)
|Method|Backbone|Pretrain|$AP$|$AP_{50}$|$AP_{75}$|Model|
|:-|:-|:-|:-|:-|:-|:-|
|LCP-50|ResNet50||29.0|46.6|30.2|

### Rainy Scene (Rainy-Cityscapes)
|Method|Backbone|Pretrain|$AP$|$AP_{50}$|$AP_{75}$|Model|
|:-|:-|:-|:-|:-|:-|:-|
|LCP-50|ResNet50||27.6|45.9|27.9|

### Underwater Scene (URP2020)
|Method|Backbone|Pretrain|$AP$|$AP_{50}$|$AP_{75}$|Model|
|:-|:-|:-|:-|:-|:-|:-|
|LCP-50|ResNet50||47.8|81.8|50.3|

## Usage
### Installing
To create a new environment, run:
```shell
conda create -n lcp python=3.10 -y
conda activate lcp
```
To install pytorch run:
```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y
```
To install mmdetection, run:
```shell
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html 

pip install yapf==0.40.1 mmdet==2.28.2 future tensorboard
```
To clone LCP, run:
```shell
git clone
cd learning-cruxes-to-push
```
### Data Preperation
The data should be orginized as follow:
```
learning-cruxes-to-push/
    data/
        city/
            annotations/
            city_images/
            foggy_city_images/
            rainy_city_images/
        urpc2020/
            annotations/
            images/
        dfui/
            images/
```
All images are converted into JPEG format, and put into folders with the suffix 'images' with no subfolders.


### Test
Here we take `LCP-50` on URPC2020 as an example.

First download our checkpoint file to `checkpoints/lcp_r50_urpc.pth`:
```shell
mkdir checkpoints
wget -P ./checkpoints/ https://
```
Then test our model (set '--cfg-options' to avoid loading pre-trained weights):
```shell
python tools/test.py \
    configs/lcp_r50_urpc.py \
    ./checkpoints/lcp_r50_urpc.pth \
    --eval bbox \
    --cfg-options model.init_cfg=None
```
### Training
Here we take a training on the  underwater scene as an example.

Fist download our pre-trained model:
```shell
wget -P ./checkpoints/ http://
```
Then train a model:
```shell
python tools/train.py \
    configs/lcp_r50_urpc.py
```

**Notes:**
- Config files of other scenes can be found in [configs/](configs/).
- Other models can be found in section [Models and Results](#models-and-results).
- For more information (e.g., about training on a custom dataset or modifying models), please refer to [MMDetection's documentation](https://mmdetection.readthedocs.io/en/v2.28.2/).