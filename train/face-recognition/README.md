# Face Recognition with ArcFace and CosFace
This repository contains the framework for training face recognition models

## Dependencies
pip install -r requirements.txt

## Dataset and prepare data for training
*Structure of dataset*

```python
face-recognition-data ------ trainset ------ Duc ------ img_1.jpg
                     |               |          |------ img_2.jpg
                     |               |          |------ ......
                     |               |          |------ img_n.jpg
                     |               |------ HDuc
                     |               |------ .....
                     |               |------ Truong
                     |------ testset
                     |------ newperson
```
*Prepare data for training*

Save preprecessed images with file name: face_224x224.npz
```python
python extract_face.py face_224x224.npz
```

## Training
### 1. Training
```python
python train.py --train \
                --loss_type arcface \
                --model model.pth \
                --plot model_his.npy   
```
***Note***: Replace --loss_type = 'cosface' if you wanna train with 'cosface' loss

### 2. Visualize

***If you want to visualize data with model after training, run the following:***

```python
python train.py --plot_simul
```

***Or you want to visualize Accuracy\Loss curves:***

```python
python train.py --plot_his
```

## Inference
```python
python infer.py --filename Vinales.jpg --loss_type arcface
```

***Note***: Replace --loss_type = 'cosface' if you want to use with 'cosface' loss and --filename with other image

## Enroll new person
```python
python enroll_newperson.py --loss_type arcface
```
***Note***: Replace --loss_type = 'cosface' if you want to use with 'cosface' loss
