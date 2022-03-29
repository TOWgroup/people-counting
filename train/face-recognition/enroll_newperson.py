
import os
import sys
import time
import numpy as np
from extract_face import extract_face
from model_exp import ConvAngular
import torchvision.transforms as transforms
import torch
from train_exp import load_data, get_embedding
import torch.nn.functional as F


def embeds_space(model, no_classes):
    train_loader = load_data(batch_size = 32)
    embeds, labels = get_embedding(model, train_loader)
    
    embeds_class = {}
    for i in range(no_classes):
        embeds_class[i] = embeds[labels == i]
    return embeds_class


def generate_embeds(model, filename):
    face = extract_face(filename)
    model = model.to('cuda').eval()
    
    face = transforms.ToTensor()(face)
    face = face.unsqueeze(0).to('cuda')
    embeds = model(face, return_embedding = True)
    norm_embeds = F.normalize(embeds).to('cpu').detach().numpy()
    return norm_embeds    


def cal_cosine(embeds_class, input_embed):
    dist_class = []
    input_embed_norm = input_embed/np.linalg.norm(input_embed)
    for i in range(len(embeds_class)):
        embeds_class_norm =  embeds_class[i] / np.linalg.norm(embeds_class[i], axis = 1, keepdims = True)
        dist_class.append(np.max(np.dot(embeds_class_norm, input_embed_norm.T)))
    index = np.argmax(dist_class)
    return index, dist_class[index]


def enroll_img(img_paths, embeds_class):
    curr_id = len(embeds_class)
    img_names = os.listdir(img_paths)
    for img_name in img_names:
        embed = generate_embeds(model, os.path.join(img_paths, img_name))
        if embeds_class.get(curr_id) is None:
            embeds_class[len(embeds_class)] = embed
        else:
            embeds_class[curr_id] = np.concatenate((embeds_class[curr_id], embed), axis = 0)
    #   Resave embeds_class
    np.save('models\\embeddings_class_256.npy', embeds_class)
    print(f'Stranger has been added to database with ID: {curr_id}')


if __name__ == "__main__":
    
    #   Load model
    model = ConvAngular(loss_type = 'arcface')
    model.load_state_dict(torch.load('models\\model_vgg16.pth'))
    
    # Generate embedding space
    embeds_class = embeds_space(model, no_classes = 10)
    np.save('models\\embeddings_class_256.npy', embeds_class)
    # Load embedding space
    embeds_class = np.load('models\\embeddings_class_256.npy', allow_pickle = True).item()
    
    
    
    """
        Enroll specific person who you wanna add to database
    """
    img_path = 'Datasets\\face-recognition-data\\newperson'
    enroll_img(img_path, embeds_class)
    
    
    
    """
        Enroll any person to database
    """
    since = time.time()
    in_embeds = generate_embeds(model, 'Datasets\\face-recognition-data\\testset\\Van.png')
    index, dist = cal_cosine(embeds_class, in_embeds)
    
    if dist > 0.95:
        print(f'Hello {index}')
        print('Time', time.time() - since)
        embeds_class[index] = np.concatenate((embeds_class[index], in_embeds.reshape(1, -1)), axis = 0)
        np.save('models\\embeddings_class_256.npy', embeds_class)
    else:
        print('Time', time.time() - since)
        embeds_class[len(embeds_class)] = in_embeds
        #   Resave embeddings class
        np.save('models\\embeddings_class_256.npy', embeds_class)
        print(f'Stranger has been added to database with ID: {len(embeds_class)-1}')
    
    
    
    """
        Predict
    """
    in_embeds = generate_embeds(model, 'Datasets\\face-recognition-data\\testset\\Van1.png')
    index, dist = cal_cosine(embeds_class, in_embeds)
    print(dist) 
    if dist > 0.9:
        print(f'Hello {index}')
    else:
        print('Stranger')

