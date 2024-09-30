import enum
import os
import sys
import torch
import torchvision
import numpy as np

from torchvision.utils import save_image
os.environ["CUDA_VISIBLE_DEVICES"]='1'
os.chdir(sys.path[0])
from model.WCGAN import Generator_Bi,Generator_TConv,Generator_Conv_linear,Discriminator
from model.utils import clear_directory,visual_result
TF_ENABLE_ONEDNN_OPTS=0
MODEL_PATH="./output/output_model/WCGAN.ckpt"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    c_dim=63
    z_dim=100
    model_G=Generator_TConv(z_dim=z_dim,c_dim=c_dim).to(DEVICE)
    #model_G= Generator_Conv_linear(generator_layer_size, z_dim, img_dim[0], img_dim[1], c_dim).to(DEVICE)
    ckpt=torch.load(MODEL_PATH)
    model_G.load_state_dict(ckpt["G"],strict=False)
    clear_directory("./output/Gen_image/")
    z_sample = torch.randn(c_dim * 1, z_dim).to(DEVICE)
    c_sample = torch.tensor(np.concatenate([np.eye(c_dim)] * 1), dtype=z_sample.dtype).to(DEVICE)
    
    model_G.eval()
    with torch.no_grad():
        z = torch.randn(c_dim, z_dim, device=DEVICE)  # Generate z for all 63 classes
        labels = torch.arange(c_dim, device=DEVICE)  # Generate labels from 0 to 62
        gen_imgs = ((model_G(z_sample, c_sample)+1)/2.0).detach().cpu()  # Generate images
        for i,gen_img in enumerate(gen_imgs):
            torchvision.utils.save_image(gen_img, f'./output/Gen_image/{i}.jpg' , nrow=10)
    
if __name__=="__main__":
    main()
