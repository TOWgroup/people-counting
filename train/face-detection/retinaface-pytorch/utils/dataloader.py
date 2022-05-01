import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image

from utils.utils import preprocess_input
import json


class DataGenerator(data.Dataset):
    def __init__(self, json_path, img_size):
        self.img_size = img_size
        self.json_path = json_path

        self.imgs_path, self.words = self.process_labels()

    def __len__(self):
        return len(self.imgs_path)

    def get_len(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        #-----------------------------------#
        #   打开图像，获取对应的标签
        #-----------------------------------#
        img = Image.open(self.imgs_path[index])
        # img=cv2.imread(self.imgs_path[index])
        # height,width,_=img.shape
        height,width=(640,640)
        annotations=np.zeros((0,5))
        labels = self.words[index]
     
        if len(labels) == 0:
            return annotations
        for i in range(len(self.words)):
            annotation = np.zeros((1, 5))
            # bbox
            annotation[0, 0] =np.max((0, self.words[i][0])) # x1
            annotation[0, 1] = np.max((0, self.words[i][1]))  # y1
            annotation[0, 2] =  np.min((width, annotation[0, 0] + np.max((0, self.words[i][2])))) # x2
            annotation[0, 3] = np.min((height,  annotation[0, 1]+ np.max((0, self.words[i][3])))) # y2
            if annotation[0,0]>0:
              annotation[0,4]=1
            else:
              annotation[0,4]=0
            if annotation[0, 2] >= annotation[0, 0] and annotation[0, 3] >= annotation[0, 1]:
              annotations = np.append(annotations, annotation, axis=0)
        target=np.array(annotations)
        img, target = self.get_random_data(img, target, [self.img_size,self.img_size])

        img = np.transpose(preprocess_input(np.array(img, np.float32)), (2, 0, 1))
        return img, target

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, image, targes, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4):
        iw, ih  = image.size
        h, w    = input_shape
        box     = targes

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = w/h * self.rand(1-jitter,1+jitter)/self.rand(1-jitter,1+jitter)
        scale = self.rand(0.25, 3.25)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            # if flip: 
            #     box[:, [0,2]] = w - box[:, [2,0,6,4,8,12,10]]
            #     box[:, [5,7,9,11,13]]     = box[:, [7,5,9,13,11]]
            
            center_x = (box[:, 0] + box[:, 2])/2
            center_y = (box[:, 1] + box[:, 3])/2
        
            box = box[np.logical_and(np.logical_and(center_x>0, center_y>0), np.logical_and(center_x<w, center_y<h))]

            box[:, 0:4][box[:, 0:4]<0] = 0
            # box[:, [0,2]][box[:, [0,2]]>w] = w
            # box[:, [1,3]][box[:, [1,3]]>h] = h
            
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
        box[:,4:-1][box[:,-1]==-1]=0
        # box[:, [0,2,4,6,8,10,12]] /= w
        # box[:, [1,3,5,7,9,11,13]] /= h
        box_data = box
        return image_data, box_data
    def process_labels(self):
      imgs_path=[]
      words=[]
      #json_path='/content/drive/MyDrive/retinaface-pytorch/train/_annotations.coco.json'
      f=open(self.json_path)
      data=json.load(f)
      isFirst=True
      labels=[]
      path=[]
      #save img_path from json file
      for i in range(len(data["images"])):
          path=data["images"][i]["file_name"]
          imgs_path.append('train/'+path)
      #save bbox
      for i in range(len(data["annotations"])):
          words.append(data["annotations"][i]['bbox'])
      print(imgs_path)
      print(words)
      return imgs_path, words
def detection_collate(batch):
    images  = []
    targets = []
    for img, box in batch:
        if len(box)==0:
            continue
        images.append(img)
        targets.append(box)
    images = np.array(images)
    return images, targets
