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
```python
python extract_face.py --preprocessing\
                       --filename face-recognition-data/trainset
```

## Training
```python
python train.py --train\
                --trainloader\
                --no_epochs\
                --batch_size\
                --learning_rate\               
                --losstype 'arcface'\                         
```

## Inference
```python
python infer.py
```

## Enroll new person
```python
python enroll_newperson.py
```
