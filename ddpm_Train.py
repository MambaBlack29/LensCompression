# %%
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from grayworldloss import grayworldloss
from UNET import UNet
from ddpm_Class_LinearSchedule import Diffusion as diffusion_linearbeta
from ddpm_Class_SigmoidSchedule import Diffusion as diffusion_sigmoidbeta
from dataset import Wide2LongDatasetAfterInitialEstimate

#Checking for GPU support
if torch.cuda.is_available():
    print("Running on GPU.\nGPU name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())
    device = torch.device('cuda:0')
    print("Device:",device)
    
else:
    device = 'cpu'
    print("Running on CPU.")

# %%
img_size = 512
batch_size = 12
# transform = transforms.Compose([transforms.Resize((img_size+img_size//4)),
#                                 transforms.RandomResizedCrop(size = (img_size, img_size), scale = (0.8, 1)),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_concatdataset = transforms.Compose([transforms.Resize((img_size, 2*img_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_estimate = transforms.Compose([transforms.Resize((img_size, img_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_aug = transforms.Compose([transforms.RandomVerticalFlip(p = 0.5),
                                    transforms.RandomHorizontalFlip(p = 0.5)])

dir_concatdataset = 'Dataset/f2fGT'
dir_estimates = 'Dataset/trainestimatesW2L'
dir_val = 'Dataset/valestimatesW2L'

upattn = [False, False, False]
downattn = [False, False, False]
lam = 0.01
chkpt_name = 'ddpm_T1000_512x512_Linear_L1_FL_0p01.pth'

dataset = Wide2LongDatasetAfterInitialEstimate(dir_concatdataset = dir_concatdataset, dir_estimate = dir_estimates, transform_concatdataset = transform_concatdataset, transform_estimate = transform_estimate,
                                               transform_aug = transform_aug)
dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)

num_epochs = 10000
model = UNet(device = device, c_in = 6, c_out = 3, time_dim = 256, upattn = upattn, downattn = downattn, img_size = img_size).to(device = device)

#Load chechpoint if found
if os.path.exists(chkpt_name):
    print("Loading Checkpoint:", chkpt_name)

    chkpt = torch.load(chkpt_name, map_location='cpu')
    model.load_state_dict(chkpt['model_state_dict'])

    del chkpt
    torch.cuda.empty_cache()

diffusion_train  = diffusion_linearbeta(noise_steps = 1000, img_size = img_size, device = device, beta_start = 1e-4, beta_end = 5e-3)
diffusion_sample = diffusion_sigmoidbeta(noise_steps = 1000, img_size = img_size, device = device, beta_start = 1e-4, beta_end = 5e-3, a = 8)
l = len(dataloader)
print('Dataset Length:', l*batch_size)

optimizer = optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = num_epochs*l)
# criterion = nn.MSELoss(reduction = 'none')  #dont do any mean, just return (x-y)**2
criterion = nn.L1Loss(reduction = 'none')
fidelity = nn.MSELoss(reduction = 'none')
# diffusion = Diffusion(img_size = img_size, device = device, noise_steps = 300, beta_start = 1e-4, beta_end = 5e-3)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model parameter Count = ", num_params)

# %%
#Training Loop
lowest_loss_per_epoch = float('inf')
for epoch in range(num_epochs):
    print("Epoch No: ", epoch)
    losses=[]
    without_fl=[]
    fl=[]
    # pbar = tqdm(dataloader)
    pbar = dataloader
    for i, images in enumerate(pbar):
        # images = images.to(device)
        # t = diffusion.sample_timesteps(n = images.shape[0]).to(device)
        # x_t, noise = diffusion.noise_images(images, t)
        # predicted_noise = model(x_t, t)
        estimate, GT = images
        estimate = estimate.float().to(device)
        GT = GT.float().to(device)

        '''Compute residual'''
        residual = (GT - estimate)  #[-1, 1] range
        '''Sample timestep'''
        t = diffusion_train.sample_timesteps(n = estimate.shape[0]).to(device)
        '''We would now want to noise the residual'''
        x_t, noise = diffusion_train.noise_images(residual, t)
        
        '''Now the model shall predict the residual noise'''
        predicted_noise = model(torch.concat([x_t, estimate], dim = 1), t)
        mse = criterion(noise, predicted_noise) #per pixel L1 or mse

        '''Compute SNR'''
        alpha_hat = diffusion_train.return_alpha_hat()
        alpha_hat = alpha_hat[t]
        snr = alpha_hat/(1 - alpha_hat) #shape: (B,)

        '''Reconstruct the image'''
        alpha_hat = alpha_hat.view(-1, 1, 1, 1)
        x0_pred = (x_t - torch.sqrt(1 - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)
        recon = x0_pred + estimate

        gamma = 0.5
        weights = 1/((snr+1)**gamma)
        weights = weights[:, None, None, None]
        loss = (weights*mse).mean()
        fidelity_loss = fidelity(recon, GT).mean()
        # loss = loss + grayworldloss(residual = residual)    #FOR INCORPORATING GRAY WORLD LOSS
        without_fl.append(loss.item())
        fl.append(lam*fidelity_loss.item())
        loss = loss + lam*fidelity_loss    #FOR INCORPORATING FIDELITY LOSS

        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    print("Non-FL Loss: ", np.mean(without_fl), "Fidelity Loss: ", np.mean(fl), "Loss: ", np.mean(losses), "Lowest loss: ", lowest_loss_per_epoch)
    if(np.mean(losses)<lowest_loss_per_epoch):    
        lowest_loss_per_epoch = np.mean(losses)
        torch.save(
            {'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            },chkpt_name)
        print('Checkpoint Saved')
        # print('Starting Validaion')
        # model.eval()
        # diffusion_sample.sample(model = model, test_dir = dir_val, transform = transform_estimate)
        # print("Validation Complete")
        # model.train()
    
    if(lowest_loss_per_epoch<0.0145):
        break
        
print('DDPM Training Complete.')
    
# %%
