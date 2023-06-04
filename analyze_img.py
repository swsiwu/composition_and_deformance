
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
from scipy.spatial import distance
from img2vec_pytorch import Img2Vec
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from collections import defaultdict

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')
#model = EfficientNet.from_pretrained('efficientnet-b0')
model = models.resnet18(pretrained=True)
layer = model._modules.get('avgpool')
model.eval()
scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(o.data.size(1)))
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding



def get_embedding(img_path):
    """
    https://github.com/christiansafka/img2vec
    
    maybe also Convolutional AutoEncoder?
    
    then Hashing, cos similarity,
    """

    
    features = get_vector(img_path)
    #print(features.shape)
    return features.unsqueeze(0)

def euc_cos_img_embedding_distance(list_of_img):
    if len(list_of_img) == 0: #######
        return 0.0 #########
    imgs = list_of_img
    embeddings = []
    for img in imgs:
        embeddings.append(get_embedding(img))
    
    embedding_pairs =  [(a, b) for idx, a in enumerate(embeddings) for b in embeddings[idx + 1:]]
    ave_cos_sim = 0.0
    ave_euclidean_distance = 0.0
    
    for a,b in embedding_pairs:
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_sim = cos(a, b)

        ave_cos_sim += cos_sim
    ave_cos_sim = float(ave_cos_sim / float(len(embedding_pairs)))

    return ave_cos_sim


def get_list_of_imgs(images_folder):
    images = []
    for img_file in os.listdir(images_folder):
        if img_file != '.DS_Store':
            img_path = images_folder + img_file
            #generate_color_histogram(img_path, img_file)
            images.append(img_path)
    return images


def get_mean_r_g_b_js(pairs):
    mean_r_js = 0.0
    mean_b_js = 0.0
    mean_g_js = 0.0
    for im_a, im_b in pairs:
        r_js_div, g_js_div, b_js_div = pair_wise_JS_divergence(im_a, im_b)
        mean_r_js += r_js_div
        mean_g_js += g_js_div
        mean_b_js += b_js_div
    print("Mean Red JS div: ", mean_r_js / len(pairs))
    print("Mean Green JS div: ", mean_g_js / len(pairs))
    print("Mean Blue  JS div: ", mean_b_js / len(pairs))
    
    
def run(folder_name):
    all_imgs = './img_result/cornell_newsroom/'+ folder_name+ '/'
    stats_path = './stats/cornell_newsroom/'+ folder_name+ '/embedding_cos.csv'
    stats_output = open(stats_path, 'w')

    for images_folder_name in os.listdir(all_imgs):
        if images_folder_name != '.DS_Store':
            images_folder = all_imgs + images_folder_name + '/'
            img_ave_cos_dist = 0.0
            line_cnt = 0
            for line_folder in os.listdir(images_folder):
                if line_folder != '.DS_Store':
                    line_cnt += 1
                    line_path = images_folder + line_folder + '/'
                    images = get_list_of_imgs(line_path)
                    pairs = [(a, b) for idx, a in enumerate(images) for b in images[idx + 1:]]
                    line_ave = euc_cos_img_embedding_distance(images)
                    img_ave_cos_dist = img_ave_cos_dist +  float(line_ave)
            if line_cnt == 0: ######
                img_ave_cos_dist = 0##########
            else:
                img_ave_cos_dist = img_ave_cos_dist / float(line_cnt)
            output_print = str(images_folder_name) + ',' + str(img_ave_cos_dist) + '\n'
            stats_output.write(output_print)
            
    stats_output.close()  
    





run('just_noun')
print('Done with just_noun')

run('original')
print('Done with original')

run('backward')
print('Done with just_noun')

poems('same_img_noun')
print('Done with same_img')

poems('permuted')
print('Done with permuted')
