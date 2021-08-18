## pix2pix

Lightweight PyTorch implementation of pix2pix (https://arxiv.org/pdf/1611.07004.pdf). Pix2pix is a popular example of a conditional GAN that solves image-translation tasks. It uses classic GAN loss as well as L1-loss to produce photo-realistic images.

## Image examples

After 10 epochs

<div align="center"><i><small>Ground truth (real) example</small></i></div>
<p align="center"><img width="250" src="https://github.com/ivankunyankin/pix2pix/blob/master/assets/real_example.png"></p>

<div align="center"><i><small>Input (sketch) example</small></i></div>
<p align="center"><img width="250" src="https://github.com/ivankunyankin/pix2pix/blob/master/assets/input_example.png"></p>

<div align="center"><i><small>Generated image (fake) example</small></i></div>
<p align="center"><img width="250" src="https://github.com/ivankunyankin/pix2pix/blob/master/assets/fake_example.png"></p>

## Installation

1. Clone the repository
``` 
git clone https://github.com/ivankunyankin/pix2pix.git
cd pix2pix 
```

2. Create an environment  and install the dependencies
``` 
python3 -m venv env 
source env/bin/activate 
pip3 install -r requirements.txt 
```

## Data

The model in this repository was trained using [Simpson Faces](https://www.kaggle.com/kostastokis/simpsons-faces) dataset from Kaggle. To be more precise using a cleaned (a bit) version of it.

I wanted to try to develop a model that will be able to generate realistic simpsons-style faces from sketch-like drawings. To do that I used OpenCV to extract contours from real images, then filtered out small ones and saved with white background trying to mimic real sketches. Certainly, there can be a way to make inputs much more similar to real sketch images. One can try to automate some sort of image distortion to do that.

To reproduce the results:
1. Download the data. [Link](https://www.kaggle.com/kostastokis/simpsons-faces)
2. Prepare the data by running the following:
```
python3 prepare_data.py
```
This script will generate input images, save them and split them by train and val sets.

### Training

In order to start training you need to run:
```
python3 train.py
```
You can play around with hyper-parameters values. You will find them in the same script.

### Tensorboard

You can watch the training process and see the intermediate results in tensorboard. Run the following:
```
tensorboard --logdir logs
```
