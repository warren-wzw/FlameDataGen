import os
import sys
os.chdir(sys.path[0])
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
import torch
import tqdm 
import numpy as np
import torch.multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')

from datetime import datetime
from PIL import Image
from model.template import DIT,RF
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from model.utils import TemplateDataset
from model.utils import load_and_cache_withlabel,get_linear_schedule_with_warmup,\
    PrintModelInfo,CaculateAcc,save_ckpt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

LR=5e-4   
STEP=10
EPOCH=300 
NUM_CLASS=10
BATCH_SIZE=128
IMAGE_SIZE=32
DEVICE_VAL = 1
DEVICE_TRAIN = 0
TENSORBOARDSTEP=500
MODEL_NAME=f"RF_{IMAGE_SIZE}.ckpt"
LAST_MODEL_NAME=f"RF_{IMAGE_SIZE}_last.ckpt"
SAVE_PATH='./output/output_model/'
PRETRAINED_MODEL_PATH=SAVE_PATH+LAST_MODEL_NAME
PRETRAINED=True if PRETRAINED_MODEL_PATH != "" and os.path.exists(PRETRAINED_MODEL_PATH) else False
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""dataset"""
train_type="train"
data_path_train=f"./dataset/src/{train_type}"
label_path=f"./dataset/label/{train_type}/{train_type}.json"
cached_file=f"./dataset/cache/{train_type}_{IMAGE_SIZE}.pt"
val_type="val"
data_path_val=f"./dataset/test/{val_type}"
cached_file_val=f"./dataset/cache/{val_type}.pt"
step=0
    
def CreateDataloader(image_path,label_path,cached_file):
    features = load_and_cache_withlabel(image_path,label_path,cached_file,image_size=IMAGE_SIZE,shuffle=True)  
    num_features = len(features)
    num_features = int(num_features)
    train_features = features[:num_features]
    dataset = TemplateDataset(features=train_features,num_instances=num_features)
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

def train_process(rf,model,event,shared_dict):
    model.train()
    global_step=0
    best_loss=100000 
    shared_dict['index'] = 0  # 共享变量初始化
    """Create dataloader"""
    # dataloader_train=CreateDataloader(data_path_train,label_path,cached_file)
    # total_steps = len(dataloader_train) * EPOCH
    """CIFAR dataset"""
    from torchvision.datasets import CIFAR10
    from torchvision import transforms
    dataset = CIFAR10(
        root='./dataset/CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
        )
    dataloader_train = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    total_steps = len(dataloader_train) * EPOCH
    """Loss function"""
    
    """Optimizer"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    """Lr"""
    scheduler = get_linear_schedule_with_warmup(optimizer, 0.1 * total_steps , total_steps)
    """tensorboard"""
    tb_writer = SummaryWriter(log_dir='./output/tflog/') 
    """Pretrain"""
    start_ep=0
    if PRETRAINED:
        ckpt = torch.load(PRETRAINED_MODEL_PATH,weights_only=True)
        start_ep = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        # optimizer.load_state_dict(ckpt['optimizer'])
        # scheduler.load_state_dict(ckpt['scheduler'])  
    torch.cuda.set_device(DEVICE_TRAIN)
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
        shared_dict['index']=epoch_index
        for step, (image,label) in enumerate(train_iterator):
            image,label= image.to(DEVICE),label.to(DEVICE)
            optimizer.zero_grad()
            loss, _ =rf.forward(image,label)
            loss.backward()
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
        if loss_sum/(step+1) < best_loss:
            best_loss = loss_sum/(step+1)
            save_ckpt(SAVE_PATH,MODEL_NAME,model,epoch_index,scheduler,optimizer)
        else:
            save_ckpt(SAVE_PATH,LAST_MODEL_NAME,model,epoch_index,scheduler,optimizer)    
        event.set()  # 在每个 epoch 结束后触发验证
        event.clear()  # 清除事件，以便下次触发

def val_process(rf,event,shared_dict):
    torch.cuda.set_device(DEVICE_VAL)
    rf.model.to(DEVICE_VAL)  # 确保模型位于正确的设备上
    rf.model.eval()
    while True:
        event.wait()
        epoch_index = shared_dict['index']
        with torch.no_grad():
            NUM=NUM_CLASS
            cond = torch.arange(0, NUM).cuda() % NUM
            #uncond = torch.ones_like(cond) * 10
            init_noise = torch.randn(NUM, 3, IMAGE_SIZE, IMAGE_SIZE).cuda()
            images = rf.sample(init_noise, cond, null_cond=None,sample_steps=STEP)
            # image sequences to gif
            gif = []
            for index,image in enumerate(images):
                if index==len(images)-1:
                    image = image * 0.5 + 0.5
                    image = image.clamp(0, 1)
                    x_as_image = make_grid(image.float(), nrow=4)
                    img = x_as_image.permute(1, 2, 0).cpu().numpy()
                    img = (img * 255).astype(np.uint8)
                    gif.append(Image.fromarray(img))
            last_img = gif[-1]
            last_img.save(f"output/output_images/sample_{epoch_index}.png")
            event.clear()
            #event.set()
            
                    
def main():
    """Define Model"""
    model=DIT(3, IMAGE_SIZE, dim=256, n_layers=10, n_heads=8, num_classes=NUM_CLASS).to(DEVICE)
    #PrintModelInfo(model)
    rf = RF(model)
    manager = mp.Manager()
    shared_dict = manager.dict() 
    event = mp.Event()
    """ Train! """
    start_time=datetime.now()
    p_train = mp.Process(target=train_process, args=(rf,model,event,shared_dict))
    p_val = mp.Process(target=val_process, args=(rf, event,shared_dict))
    p_train.start()
    p_val.start()
    p_train.join()
    p_val.join()
    end_time=datetime.now()
    print("Training consume :",(end_time-start_time)/60,"minutes")
    
if __name__=="__main__":
    mp.set_start_method('spawn')
    main()
