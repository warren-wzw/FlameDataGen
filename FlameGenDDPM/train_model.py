import os
import sys
os.chdir(sys.path[0])
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  
import torch
import tqdm 
import torch.nn as nn
import numpy as np
import torch.optim as optim

from datetime import datetime
from model.DDPM import UNet,GaussianDiffusionTrainer,UNet_self
from torch.utils.data import DataLoader
from model.utils import FireDataset
from torchvision.utils import save_image
from model.utils import load_and_cache_withlabel,get_linear_schedule_with_warmup,\
    PrintModelInfo,GradualWarmupScheduler \

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

LR=1e-4  
EPOCH=200 
CLASS_NUM=63
BATCH_SIZE=2
TENSORBOARDSTEP=500
IMAGE_SIZE=128
MODEL_NAME=f"DDPM_{IMAGE_SIZE}_flame.ckpt"
LAST_MODEL_NAME=f"DDPM_{IMAGE_SIZE}_flame_last.ckpt"
PRETRAINED_MODEL_PATH=f"./output/output_model/{LAST_MODEL_NAME}"
PRETRAINED=True if PRETRAINED_MODEL_PATH != "" and os.path.exists(PRETRAINED_MODEL_PATH) else False
SAVE_PATH='./output/output_model/'
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""dataset"""
train_type="train"
data_path_train=f"./dataset/src/train"
label_path_train=f"./dataset/label/{train_type}/{train_type}.json"
cached_file=f"./dataset/cache/{train_type}_{IMAGE_SIZE}_flame.pt"
val_type="val"
data_path_val=f"./dataset/test/test"
cached_file_val=f"./dataset/cache/{val_type}.pt"

def CreateDataloader(image_path,label_path,cached_file):
    features = load_and_cache_withlabel(image_path,label_path,cached_file,image_size=IMAGE_SIZE,shuffle=True)  
    num_features = len(features)
    num_features = int(num_features)
    train_features = features[:num_features]
    dataset = FireDataset(features=train_features,num_instances=num_features)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

def main():
    global_step=0
    """Define Model"""
    model = UNet(TimeStep=500, num_labels=CLASS_NUM, channel=128, ch_mult=[1, 2, 2, 2],
                    num_res_blocks=2,dropout=0.15).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    trainer = GaussianDiffusionTrainer(model, 1e-4, 0.028, 500).to(DEVICE)
    #PrintModelInfo(model)
    """Create dataloader"""
    dataloader_train=CreateDataloader(data_path_train,label_path_train,cached_file)
    """CIFAR dataset"""
    # from torchvision.datasets import CIFAR10
    # from torchvision import transforms
    # dataset = CIFAR10(
    #     root='./dataset/CIFAR10', train=True, download=True,
    #     transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    #     )
    # dataloader_train = DataLoader(
    #     dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    
    """Optimizer"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,weight_decay=1e-4)
    """Lr"""
    total_steps = len(dataloader_train) * EPOCH
    #scheduler=get_linear_schedule_with_warmup(optimizer,0.1*total_steps,total_steps)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=EPOCH, eta_min=0, last_epoch=-1)
    scheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=2.5, \
        warm_epoch=EPOCH // 10, after_scheduler=cosineScheduler)
    """tensorboard"""
    tb_writer = SummaryWriter(log_dir='./output/tflog/') 
    """Pretrain"""
    start_ep=0
    if PRETRAINED:
        ckpt = torch.load(PRETRAINED_MODEL_PATH)
        start_ep = ckpt['epoch']
        model.load_state_dict(ckpt['model'],strict=False)
        # optimizer.load_state_dict(ckpt['optimizer'])
        # scheduler.load_state_dict(ckpt['scheduler'])
    """ Train! """
    model.train()
    best_loss=1000000 
    start_time=datetime.now()
    print("  ************************ Running training ***********************")
    print("  Num Epochs = ", EPOCH)
    print("  Batch size per node = ", BATCH_SIZE)
    #print("  Num examples = ", dataloader_train.sampler.data_source.num_instances)
    print(f"  Pretrained Model is {PRETRAINED_MODEL_PATH}")
    print(f"  Save Model as {SAVE_PATH+MODEL_NAME}")
    print("  ****************************************************************")
    for epoch_index in range(start_ep,EPOCH):
        loss_sum=0
        sum_test_accuarcy=0
        torch.cuda.empty_cache()
        train_iterator = tqdm.tqdm(dataloader_train, initial=0,desc="Iter", disable=False)
        for step, (image,label) in enumerate(train_iterator):
            image,label= image.to(DEVICE),label.to(DEVICE)+1
            """save images"""
            # for i,image_ in enumerate(image):
            #      save_image(image_, os.path.join("./dataset/CIFAR10/images",  f"{step*BATCH_SIZE+i}.png"), nrow=1)
            optimizer.zero_grad()
            if np.random.rand() < 0.1:
                label = torch.zeros_like(label).to(DEVICE)
            loss = trainer(image, label).sum() / (image.shape[0]) ** 2.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),1.)
            optimizer.step()
            """cal loss and acc"""
            loss_sum=loss_sum+loss.item()
            """ tensorbooard """
            current_lr= scheduler.get_last_lr()[0]
            if  global_step % TENSORBOARDSTEP== 0 and tb_writer is not None:
                tb_writer.add_scalar('train/lr', current_lr, global_step=global_step)
                tb_writer.add_scalar('train/loss', loss.item(), global_step=global_step)
            """show progress bar"""
            train_iterator.set_description('Epoch=%d, Acc= %3.3f %%,loss=%.6f, lr=%9.7f' 
                                           % (epoch_index,(sum_test_accuarcy/(step+1))*100, loss_sum/(step+1), current_lr))
            global_step=global_step+1
        scheduler.step()
        """save the best"""
        if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
        if loss_sum/(step+1) < best_loss:
            best_loss = loss_sum/(step+1)
            torch.save({'epoch': epoch_index + 1,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler':scheduler.state_dict()},
                                '%s%s' % (SAVE_PATH,MODEL_NAME))
            print("->Saving model {} at {}".format(SAVE_PATH+MODEL_NAME, 
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        else:
            torch.save({'epoch': epoch_index + 1,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler':scheduler.state_dict()},
                                '%s%s' % (SAVE_PATH,LAST_MODEL_NAME)) 
            print("->Saving model {} at {}".format(SAVE_PATH+LAST_MODEL_NAME, 
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")))  
    end_time=datetime.now()
    print("Training consume :",(end_time-start_time)/60,"minutes")
    
if __name__=="__main__":
    main()
