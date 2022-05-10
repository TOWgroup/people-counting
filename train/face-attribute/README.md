# Face Attribute
This repository contains the framework for training face attribute with CelebA dataset

## Dataset and prepare data for training
*Structure of dataset*
```python
img_align_celeba ------- img_align_celeba ------- 000001.jpg
                |                        |------- 000002.jpg
                |                        |------- 000002.jpg
                |                        |------- .......
                |                        |------- 202599.jpg
                |                                        
                |------- list_attr_celeba.csv            
                |------- list_bbox_celeba.csv         
                |------- list_eval_partition.csv             
                |------- list_landmarks_align_celeba.csv
```

*Attributes in list_attr_celeba.csv file*

['5_o_Clock_Shadow',
 'Arched_Eyebrows',
 'Attractive',
 'Bags_Under_Eyes',
 'Bald',
 'Bangs',
 'Big_Lips',
 'Big_Nose',
 'Black_Hair',
 'Blond_Hair',
 'Blurry',
 'Brown_Hair',
 'Bushy_Eyebrows',
 'Chubby',
 'Double_Chin',
 'Eyeglasses',
 'Goatee',
 'Gray_Hair',
 'Heavy_Makeup',
 'High_Cheekbones',
 'Male',
 'Mouth_Slightly_Open',
 'Mustache',
 'Narrow_Eyes',
 'No_Beard',
 'Oval_Face',
 'Pale_Skin',
 'Pointy_Nose',
 'Receding_Hairline',
 'Rosy_Cheeks',
 'Sideburns',
 'Smiling',
 'Straight_Hair',
 'Wavy_Hair',
 'Wearing_Earrings',
 'Wearing_Hat',
 'Wearing_Lipstick',
 'Wearing_Necklace',
 'Wearing_Necktie',
 'Young']
 
## Prepare data for training
 
 Load data and split for training
```python
python main.py --filepath path_to_list_attr_celeba.csv --data_process
```

## Training
```python
python main.py --train path_to_save_model
```

## Testing
```python
python main.py --pred path_to_save_image_result.png
```
