import os
import sys
import re
import torch
import numpy as np
import random
import tqdm
import json
import shutil
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image,ImageEnhance
from torch.utils.data import  Dataset
from torch.optim.lr_scheduler import LambdaLR

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def PrintModelInfo(model):
    """Print the parameter size and shape of model detail"""
    total_params = 0
    for name, param in model.named_parameters():
        num_params = torch.prod(torch.tensor(param.shape)).item() * param.element_size() / (1024 * 1024)  # 转换为MB
        print(f"{name}: {num_params:.4f} MB, Shape: {param.shape}")
        total_params += num_params
    print(f"Total number of parameters: {total_params:.4f} MB")     

def min_max_normalize(image):
    np_image = np.array(image).astype(np.float32)
    np_image = (np_image - np.min(np_image)) / (np.max(np_image) - np.min(np_image))
    return torch.tensor(np_image)

def clear_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
            
def visual_result(input,filename):
    if len(input.shape)==4:
        np_image = input[0].cpu().permute(1,2,0).numpy()  # 将通道维度移到最后
    elif len(input.shape)==3:
        np_image = input.cpu().permute(1,2,0).numpy()  # 将通道维度移到最后
    if np_image.min()<0:    
        np_image = np_image * 0.5 + 0.5  # 假设图像已归一化为[-1, 1]
    plt.imshow(np_image)
    plt.axis('off')
    plt.savefig(filename)  # 在绘制图像后保存  
    
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(size=(128, 128), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  
    ])
    image = Image.open(image).convert('RGB')
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.8)  # 增强对比度的因子，可以调整这个值
    image=transform(image)
    return image

def CaculateAcc():
    print()

def sort_key(filename):
    match = re.search(r'(\d+)_(\d+)_(\d+)\.', filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    else:
        return float('inf'), float('inf'), float('inf')  
         
def load_and_cache_withlabel(image_path,label_path,cache_file,shuffle=False):
    if cache_file is not None and os.path.exists(cache_file):
        print("Loading features from cached file ", cache_file)
        features = torch.load(cache_file)
    else:
        print("Creating features from dataset at ", image_path,label_path)
        images,labels = [],[]
        for img_name in os.listdir(image_path):
            img_path = os.path.join(image_path, img_name)
            images.append(img_path)
        images=sorted(images,key=sort_key)
        with open(label_path,'r') as json_file:
             for i,line in enumerate(json_file):
                labels.append(json.loads(line))
        features = []
        def get_label_data(label):
            file=label["file:"]
            match = re.match(r'(\d+)_(\d+)', label["status"])
            if match:
                n = int(match.group(1))
                m = int(match.group(2))
                result = 7 * (n - 1) + m-1
            status=result
            O2=label["O2"]
            O2_CO2=label["O2_CO2"]
            CH4=label["CH4"]
            O2_CH4=label["CH4_CO2"]
            return file,status,O2,O2_CO2,CH4,O2_CH4        
              
        total_iterations = len(images)  # 设置总的迭代次数  
        for image_path,label in tqdm.tqdm(zip(images,labels),total=total_iterations):
            #processed_image=preprocess_image(image_path)
            processed_image=preprocess_image(image_path) #only cgan
            #visual_result(processed_image,"output.jpg")
            file,status,O2,O2_CO2,CH4,O2_CH4 =get_label_data(label)
            feature = {
                "image": processed_image,
                "file": file,
                "label":status,
                "O2":O2,
                "O2_CO2":O2_CO2,
                "CH4":CH4,
                "O2_CH4":O2_CH4
            }
            features.append(feature)
 
        if shuffle:
            random.shuffle(features)
        if not os.path.exists(cache_file):
            print("Saving features into cached file ", cache_file)
            torch.save(features, cache_file)
    return features

def load_and_cache_withlabel_bed(image_path,cache_file,image_size=0,shuffle=False):
    if cache_file is not None and os.path.exists(cache_file):
        print("Loading features from cached file ", cache_file)
        features = torch.load(cache_file)
    else:
        print("Creating features from dataset at ", image_path)
        images,labels = [],[]
        for img_name in os.listdir(image_path):
            img_path = os.path.join(image_path, img_name)
            images.append(img_path)
        images=sorted(images,key=sort_key)
        features = []               
        total_iterations = len(images)  # 设置总的迭代次数  
        for image_path in tqdm.tqdm(images,total=total_iterations):
            processed_image=preprocess_image(image_path) #only cgan
            #visual_result(processed_image,"output.jpg")
            feature = {
                "image": processed_image,
                "label":1,
            }
            features.append(feature)
 
        if shuffle:
            random.shuffle(features)
        if not os.path.exists(cache_file):
            print("Saving features into cached file ", cache_file)
            torch.save(features, cache_file)
    return features

class FireDataset(Dataset):
    def __init__(self,features,num_instances):
        self.feature=features
        self.num_instances=num_instances
    
    def __len__(self):
        return int(self.num_instances)
    
    def __getitem__(self, index):
        feature = self.feature[index]
        image=feature["image"]
        label=feature["label"]
        return image,label

def save_checkpoint(obj, save_path, is_best=False, max_keep=None):
    # save checkpoint
    torch.save(obj, save_path)

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, 'latest_checkpoint')

    save_path = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_path + '\n'] + ckpt_list
    else:
        ckpt_list = [save_path + '\n']

    # if max_keep is not None:
    #     for ckpt in ckpt_list[max_keep:]:
    #         ckpt = os.path.join(save_dir, ckpt[:-1])
    #         if os.path.exists(ckpt):
    #             os.remove(ckpt)
    #     ckpt_list[max_keep:] = []

    with open(list_path, 'w') as f:
        f.writelines(ckpt_list)

    # copy best
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, 'best_model.ckpt'))

def load_checkpoint(ckpt_dir_or_file, map_location=None, load_best=False):
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')
        else:
            with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint')) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint succeeds! Copy variables from % s!' % ckpt_path)
    return ckpt
          
