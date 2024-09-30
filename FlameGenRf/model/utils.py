import os
import sys
import re
import torch
import tqdm
import shutil
import random
import json
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import  Dataset
from PIL import Image,ImageEnhance     
from torch.optim.lr_scheduler import LambdaLR

"""Model info"""
def PrintModelInfo(model):
    """Print the parameter size and shape of model detail"""
    total_params = 0
    model_parments = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for name, param in model.named_parameters():
        num_params = torch.prod(torch.tensor(param.shape)).item() * param.element_size() / (1024 * 1024)  # 转换为MB
        print(f"{name}: {num_params:.4f} MB, Shape: {param.shape}")
        total_params += num_params
    print(f"----->Traing parments {model_parments/1e6:4f}M") 
    print(f"---->Model Size: {total_params:.4f} MB")    

"""Lr"""
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

"""Dataset"""   
def min_max_normalize(image):
    np_image = np.array(image).astype(np.float32)
    np_image = (np_image - np.min(np_image)) / (np.max(np_image) - np.min(np_image))
    return torch.tensor(np_image)

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
    
def preprocess_image(image,image_size=0):
    transform = transforms.Compose([
        transforms.Resize(size=(image_size, image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    ])
    image = Image.open(image).convert('RGB')
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.8)  # 增强对比度的因子，可以调整这个值
    image=transform(image)
    return image

def sort_key(filename):
    match = re.search(r'(\d+)_(\d+)_(\d+)\.', filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    else:
        return float('inf'), float('inf'), float('inf')

def load_and_cache_withlabel(image_path,label_path,cache_file,image_size=0,shuffle=False):
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
            processed_image=preprocess_image(image_path,image_size=image_size) 
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

class TemplateDataset(Dataset):
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

"""Save and load model"""
from datetime import datetime
def save_ckpt(save_path,model_name,model,epoch_index,scheduler,optimizer):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({'epoch': epoch_index + 1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler':scheduler.state_dict()},
                        '%s%s' % (save_path,model_name))
    print("->Saving model {} at {}".format(save_path+model_name, 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
"""Evaluate model"""
def CaculateAcc(output,label):
    print()
   
