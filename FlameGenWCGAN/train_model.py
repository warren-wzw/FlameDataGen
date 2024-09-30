import os
import sys
from tkinter import E
import tqdm
import torch
import torchvision
import matplotlib.pyplot as plt
import shutil
import numpy as np
os.chdir(sys.path[0])
os.environ["CUDA_VISIBLE_DEVICES"]='1'
from torch.utils.data import (DataLoader)
from datetime import datetime
from model.WCGAN import Generator_Bi,Generator_TConv,Generator_Conv_linear,Discriminator,GeneratorCGAN,DiscriminatorCGAN
from model.utils import FireDataset,load_and_cache_withlabel,get_linear_schedule_with_warmup, \
    PrintModelInfo,clear_directory,visual_result,load_and_cache_withlabel_bed
from torch.autograd import grad
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
TF_ENABLE_ONEDNN_OPTS=0
BATCH_SIZE=80
EPOCH=500
LR_G=10
LR_D=10
LR=1e-5
NUM_CLASS=1
PRETRAINED_MODEL_PATH="./output/output_model/WCGAN_BED.ckpt"
TensorBoardStep=500
SAVE_MODEL='./output/output_model/'
"""dataset"""
train_type="train"
#image_path_train=f"./dataset/image/{train_type}"
image_path_train=f"./dataset/bedroom"
label_path_train=f"./dataset/label/{train_type}/{train_type}.json"
cached_file=f"./dataset/cache/{train_type}_bed.pt"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED=False
if PRETRAINED_MODEL_PATH != "" and os.path.exists(PRETRAINED_MODEL_PATH):
    PRETRAINED=True 
    

def gradient_penalty(f, real, fake, mode):
    device = real.device

    def _gradient_penalty(f, real, fake=None):
        def _interpolate(a, b=None):
            if b is None:   # interpolation in DRAGAN
                beta = torch.rand(a.size()).to(device)
                b = a + 0.5 * a.std() * beta
            shape = [a.size(0)] + [1] * (a.dim() - 1)
            alpha = torch.rand(shape).to(device)
            inter = a + alpha * (b - a)
            return inter

        x = torch.tensor(_interpolate(real, fake), requires_grad=True)
        pred = f(x)
        if isinstance(pred, tuple):
            pred = pred[0]
        g = grad(pred, x, grad_outputs=torch.ones(pred.size()).to(device), create_graph=True)[0].view(x.size(0), -1)
        gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

        return gp

    if mode == 'wgan-gp':
        gp = _gradient_penalty(f, real, fake)
    elif mode == 'dragan':
        gp = _gradient_penalty(f, real)
    elif mode == 'none':
        gp = torch.tensor(0.0).to(device)
    else:
        raise NotImplementedError

    return gp

def CreateDataloader(image_path,label_path,cached_file):
    features = load_and_cache_withlabel(image_path,label_path,cached_file,shuffle=True)  
    num_features = len(features)
    num_train = int(1* num_features)
    train_features = features[:num_train]
    dataset = FireDataset(features=train_features,num_instances=num_train)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

def CreateDataloader_bed(image_path,cached_file):
    features = load_and_cache_withlabel_bed(image_path,cached_file,image_size=128,shuffle=True)  
    num_features = len(features)
    num_features = int(num_features)
    train_features = features[:num_features]
    dataset = FireDataset(features=train_features,num_instances=num_features)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

def Dynamic_Train(mean_loss_D, mean_loss_G, now_lossd, now_lossg, DCycleNum, GCycleNum, DThread_Num=5,GThread_Num=3):
    D_changed=False
    G_changed=False
    """D"""
    if now_lossd - mean_loss_D >= 0:
        if  now_lossd - mean_loss_D < 0.5:
            if not D_changed:
                DCycleNum += 1
                D_changed=True
        else:
            if not D_changed:
                DCycleNum -= 1
                D_changed=True
            if not G_changed:
                GCycleNum += 1
                G_changed=True
    if  now_lossd - mean_loss_D < 0:
        if now_lossd - mean_loss_D < -0.5:
            if not D_changed:
                DCycleNum -= 1
                D_changed=True
            if not G_changed:
                GCycleNum += 1
                G_changed=True
    """G"""
    if now_lossg - mean_loss_G >= 0:
        if now_lossg - mean_loss_G < 0.5:
            if not G_changed:
                GCycleNum += 1
                G_changed=True
        else:
            if not D_changed:
                DCycleNum += 1
                D_changed=True
            if not G_changed:
                GCycleNum += 1
                G_changed=True
    if now_lossg - mean_loss_G < 0:
        if now_lossg - mean_loss_G < -0.5:
            if not D_changed:
                DCycleNum += 1
                D_changed=True
            if not G_changed:
                GCycleNum += 1
                G_changed=True
    """balance"""
    DCycleNum, GCycleNum = max(1, DCycleNum), max(0, GCycleNum)
    DCycleNum, GCycleNum = min(DThread_Num, DCycleNum), min(GThread_Num, GCycleNum)
    if DCycleNum == GCycleNum:
        DCycleNum = 1
        GCycleNum = 1  
    return DCycleNum, GCycleNum
            
def main():
    c_dim=NUM_CLASS
    z_dim=100
    global_step=0 
    GCycleNum=1
    DCycleNum=1
    loss_queue_D=[]
    loss_queue_G=[]
    start_ep=0
    model_G=Generator_TConv(z_dim=z_dim, c_dim=c_dim).to(DEVICE)
    model_D=DiscriminatorCGAN(x_dim=3, c_dim=c_dim, norm='none', weight_norm="spectral_norm").to(DEVICE)
    PrintModelInfo(model_G)
    print()
    PrintModelInfo(model_D)
    dataloader_train=CreateDataloader_bed(image_path_train,cached_file)
    total_steps = len(dataloader_train) * EPOCH
    clear_directory("./output/output_images/")
    """optimizer"""
    optimizer_G = torch.optim.RMSprop(model_G.parameters(), lr=LR*LR_G)
    optimizer_D = torch.optim.RMSprop(model_D.parameters(), lr=LR*LR_D)
    """Lr"""
    scheduler_G = get_linear_schedule_with_warmup(optimizer_G, 0.1 * total_steps , total_steps)
    scheduler_D = get_linear_schedule_with_warmup(optimizer_D, 0.1 * total_steps , total_steps)
    if PRETRAINED:
        ckpt = torch.load(PRETRAINED_MODEL_PATH)
        start_ep = ckpt['epoch']
        model_D.load_state_dict(ckpt['D'])
        model_G.load_state_dict(ckpt['G'])
        optimizer_D.load_state_dict(ckpt['d_optimizer'])
        optimizer_G.load_state_dict(ckpt['g_optimizer'])
        scheduler_D.load_state_dict(ckpt['scheduler_D'])
        scheduler_G.load_state_dict(ckpt['scheduler_G'])
    """ Train! """
    print("  ************************ Running training ***********************")
    print("  Num Epochs = ", EPOCH)
    print("  Batch size per node = ", BATCH_SIZE)
    print("  Num examples = ", dataloader_train.sampler.data_source.num_instances)
    print(f"  Pretrained Model is {PRETRAINED_MODEL_PATH}")
    print(f"  Save Model as {SAVE_MODEL}")
    print("  ****************************************************************")
    tb_writer = SummaryWriter(log_dir='./output/tflog/')
    z_sample = torch.randn(c_dim * 1, z_dim).to(DEVICE)
    c_sample = torch.tensor(np.concatenate([np.eye(c_dim)] * 1), dtype=z_sample.dtype).to(DEVICE)
    for epoch_index in range(start_ep,EPOCH):
        loss_sumd=0
        loss_sumg=0
        torch.cuda.empty_cache()
        train_iterator = tqdm.tqdm(dataloader_train, initial=0,desc="Iter", disable=False)
        for step, (image,label) in enumerate(train_iterator):
            model_G.train()
            model_D.train()
            image,label= image.to(DEVICE),label.to(DEVICE)-1
            z = torch.randn(len(image), z_dim, device=DEVICE)
            labels = torch.tensor(np.eye(c_dim)[label.cpu().numpy()], dtype=z.dtype).to(DEVICE)
            """train model_D"""
            for i in range(DCycleNum):
                optimizer_D.zero_grad()
                fake_image=model_G(z, labels).detach()
                real_validity  = model_D(image, labels)
                fake_validity  = model_D(fake_image, labels)
                gp=gradient_penalty(model_D, image, fake_image,mode="none")
                d_loss = - torch.mean(real_validity)+torch.mean(fake_validity)+1*gp
                model_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()
            """train G model"""
            for i in range(GCycleNum):
                z = torch.randn(len(image), z_dim, device=DEVICE)
                optimizer_G.zero_grad()
                gen_imgs = model_G(z, labels)
                fake_validity= model_D(gen_imgs, labels)
                g_loss = -torch.mean(fake_validity)
                model_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()
            """training detail"""    
            current_lr_G= scheduler_G.get_last_lr()[0]
            current_lr_R= scheduler_D.get_last_lr()[0]
            scheduler_G.step()
            scheduler_D.step()
            loss_sumg=loss_sumg+g_loss.item()
            loss_sumd=loss_sumd+d_loss.item()
            """tqdm"""
            train_iterator.set_description('Epoch=%d, loss_G=%.6f, loss_D=%.6f, lr_G=%9.7f,lr_D=%9.7f,D_NUM=%d,G_NUM=%d'% (
                epoch_index, loss_sumg/(step+1), loss_sumd/(step+1),current_lr_G,current_lr_R,DCycleNum,GCycleNum))
            """ tensorboard """
            if  global_step % TensorBoardStep== 0 and tb_writer is not None:
                tb_writer.add_scalar('train_G/lr', scheduler_G.get_last_lr()[0], global_step=global_step)
                tb_writer.add_scalar('train_G/loss', g_loss.item(), global_step=global_step)
            if  global_step % TensorBoardStep== 0 and tb_writer is not None:
                tb_writer.add_scalar('train_D/lr', scheduler_D.get_last_lr()[0], global_step=global_step)
                tb_writer.add_scalar('train_D/loss', d_loss.item(), global_step=global_step)
            global_step+=1
            model_G.eval()
            with torch.no_grad():
                z = torch.randn(c_dim, z_dim, device=DEVICE)  # Generate z for all 63 classes
                gen_imgs = ((model_G(z_sample, c_sample)+1)/2.0).detach().cpu()  # Generate images
                torchvision.utils.save_image(gen_imgs, './output/output_images/test.jpg' , nrow=10)
        """cal averge loss"""
        loss_queue_D.append(loss_sumd/(step+1))
        if len(loss_queue_D) > 5:
            loss_queue_D.pop(0) 
        loss_queue_G.append(loss_sumg/(step+1))
        if len(loss_queue_G) > 5:
            loss_queue_G.pop(0)  
        mean_loss_D = sum(loss_queue_D) / len(loss_queue_D) if loss_queue_D else 0.0
        mean_loss_G = sum(loss_queue_G) / len(loss_queue_G) if loss_queue_G else 0.0
        #DCycleNum,GCycleNum=Dynamic_Train(mean_loss_D,mean_loss_G,loss_sumd/(step+1),loss_sumg/(step+1),DCycleNum,GCycleNum)
        """save model"""
        torch.save({'epoch':  epoch_index + 1,
                              'D': model_D.state_dict(),
                              'G': model_G.state_dict(),
                              'd_optimizer': optimizer_D.state_dict(),
                              'g_optimizer': optimizer_G.state_dict(),
                              'scheduler_D':scheduler_D.state_dict(),
                              'scheduler_G':scheduler_G.state_dict()},
                              '%sWCGAN_BED.ckpt' % (SAVE_MODEL)) 
        
if __name__ == "__main__":
    main()
