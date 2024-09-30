import os
import sys
from tkinter import E
os.chdir(sys.path[0])
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch
import numpy as np

from PIL import Image
from model.template import DIT,RF
from torchvision.utils import make_grid
from model.utils import load_and_cache_withlabel,PrintModelInfo,CaculateAcc
STEP=10
EPOCH=500
IMAGE_SIZE=32
NUM_CLASS=10
MODEL_PATH="./output/output_model/RF_32.ckpt"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    """Define Model"""
    model=DIT(3, IMAGE_SIZE, dim=256, n_layers=10, n_heads=8, num_classes=NUM_CLASS).to(DEVICE)
    PrintModelInfo(model)
    rf = RF(model)
    ckpt = torch.load(MODEL_PATH)
    model.load_state_dict(ckpt['model'])
    PrintModelInfo(model)
    print("model load weight done.")
    model.eval()
    with torch.no_grad():
        for epoch in range(EPOCH):
            NUM=NUM_CLASS
            cond = torch.arange(0, NUM).cuda() % NUM
            init_noise = torch.randn(NUM, 3, IMAGE_SIZE, IMAGE_SIZE).cuda()
            images = rf.sample(init_noise, cond, null_cond=None,sample_steps=STEP)
            # image sequences to gif
            for index,image in enumerate(images):
                if index==len(images)-1:
                    image = image * 0.5 + 0.5
                    image = image.clamp(0, 1)
                    for i,image_ in enumerate(image):
                        x_as_image = make_grid(image_.float(), nrow=1)
                        img = x_as_image.permute(1, 2, 0).cpu().numpy()
                        img = (img * 255).astype(np.uint8)
                        image=Image.fromarray(img)
                        image.save(f"output/output_images/sample_{63*(epoch)+i}.png")
                        print(f"output/output_images/sample_{63*(epoch)+i}.png")
    
if __name__=="__main__":
    main()