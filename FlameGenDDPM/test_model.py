import os
import sys
from tracemalloc import start
os.chdir(sys.path[0])
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import torch

from datetime import datetime
from model.DDPM import UNet,GaussianDiffusionSampler,UNet_self
from torchvision.utils import save_image
from model.utils import load_and_cache_withlabel,get_linear_schedule_with_warmup,\
    PrintModelInfo,CaculateAcc
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE=128
CLASS_NUM=63
BATCH_SIZE=1
MODEL_NAME="DDPM_128_flame_last.ckpt"
MODEL_PATH=f"./output/output_model/{MODEL_NAME}"


def main():
    num_labels = list(range(43, 44))
    # load model and evaluate
    with torch.no_grad():
        for num in range(1):
            step = int( BATCH_SIZE// len(num_labels))
            labelList = []
            k = [x - 1 for x in num_labels]
            for i in range(1,BATCH_SIZE + 1):
                label = (i - 1) // step + 1  # 计算当前的标签
                label = min(label, len(num_labels))  # 防止超出标签范围
                labelList.append(torch.ones(size=[1]).long() * label)
                        
            labels = torch.cat(labelList, dim=0).long().to(DEVICE) 
            print("labels: ", labels)
            model = UNet(TimeStep=500, num_labels=CLASS_NUM, channel=128, ch_mult=[1, 2, 2, 2],
                        num_res_blocks=2,dropout=0.15).to(DEVICE)
            model.load_state_dict(torch.load(MODEL_PATH)['model'])   
            print("model load weight done.")
            model.eval()
            sampler = GaussianDiffusionSampler(model, 2e-4, 0.028, 500,w =1.8).to(DEVICE)
            noisyImage = torch.randn(
                size=[BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE], device=DEVICE)
            start_time=datetime.now()
            sampledImgs = sampler(noisyImage, labels)
            end_time=datetime.now()
            print("inference use ",end_time-start_time,"s")
            save_image(sampledImgs, os.path.join("./output/output_images/",  f"Output.png"), nrow=5)
            # for i,sampledImg in enumerate(sampledImgs):
            #         save_image(sampledImg, os.path.join("./output/Gen",  f"{num*BATCH_SIZE+i}.png"), nrow=1)

if __name__=="__main__":
    main()