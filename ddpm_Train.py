# %%
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from grayworldloss import grayworldloss
from UNET import UNet
from ddpm_Class_LinearSchedule import Diffusion as diffusion_linearbeta
from ddpm_Class_SigmoidSchedule import Diffusion as diffusion_sigmoidbeta
from ddpm_Class_CosineSchedule import Diffusion as diffusion_cosinebeta
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
chkpt_name = 'checkpoints/ddpm_T1000_512x512_Linear_L2_ssim_1.pth'

dataset = Wide2LongDatasetAfterInitialEstimate(dir_concatdataset = dir_concatdataset, dir_estimate = dir_estimates, transform_concatdataset = transform_concatdataset, transform_estimate = transform_estimate,
                                               transform_aug = transform_aug)
dataloader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)

num_epochs = 3500
model = UNet(device = device, c_in = 6, c_out = 3, time_dim = 256, upattn = upattn, downattn = downattn, img_size = img_size).to(device = device)

diffusion_train  = diffusion_linearbeta(noise_steps = 1000, img_size = img_size, device = device, beta_start = 1e-4, beta_end = 5e-3)
# diffusion_train = diffusion_cosinebeta(noise_steps=1000, img_size=img_size, device=device, s=0.008)
diffusion_sample = diffusion_sigmoidbeta(noise_steps = 1000, img_size = img_size, device = device, beta_start = 1e-4, beta_end = 5e-3, a = 8)
l = len(dataloader)
print('Dataset Length:', l*batch_size)

optimizer = optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = num_epochs*l)
criterion = nn.MSELoss(reduction = 'none')  #dont do any mean, just return (x-y)**2
# criterion = nn.L1Loss(reduction = 'none')
fidelity = nn.MSELoss(reduction = 'none')
# diffusion = Diffusion(img_size = img_size, device = device, noise_steps = 300, beta_start = 1e-4, beta_end = 5e-3)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model parameter Count = ", num_params)

start_epoch = 0
lowest_loss_per_epoch = float('inf')

if os.path.exists(chkpt_name):
    print("Loading Checkpoint:", chkpt_name)
    chkpt = torch.load(chkpt_name, map_location='cpu')
    model.load_state_dict(chkpt['model_state_dict'])
    optimizer.load_state_dict(chkpt['optimizer_state_dict'])
    start_epoch = chkpt['epoch'] + 1
    lowest_loss_per_epoch = chkpt['lowest_loss_per_epoch']
    scheduler.load_state_dict(chkpt['scheduler_state_dict'])

    del chkpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %%
#Training Loop
for epoch in range(start_epoch, num_epochs):
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

        '''SSIM loss'''
        loss_ssim = 1 - ssim((recon+1)/2, (GT+1)/2, data_range=1.0, reduction='elementwise_mean')
        eta = 0.84

        # gamma = 0.5
        # weights = 1/((snr+1)**gamma)
        # weights = weights / weights.sum()
        # weights = weights[:, None, None, None]
        # loss = (1-eta)*(weights*mse).mean() + eta*loss_ssim

        loss = (1-eta)*(mse.mean()) + eta*loss_ssim
        # fidelity_loss = fidelity(recon, GT).mean()
        # loss = loss + grayworldloss(residual = residual)    #FOR INCORPORATING GRAY WORLD LOSS
        # without_fl.append(loss.item())
        # fl.append(lam*fidelity_loss.item())
        # loss = loss + lam*fidelity_loss    #FOR INCORPORATING FIDELITY LOSS

        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    epoch_loss = float(np.mean(losses))
    save_best = epoch_loss < lowest_loss_per_epoch
    if save_best:
        lowest_loss_per_epoch = epoch_loss

    print("Loss:", epoch_loss, "Lowest loss:", lowest_loss_per_epoch)
    # print("Non-FL Loss: ", np.mean(without_fl), "Fidelity Loss: ", np.mean(fl), "Loss: ", epoch_loss, "Lowest loss: ", lowest_loss_per_epoch)

    if save_best or epoch == 2000:
        chkpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'lowest_loss_per_epoch': lowest_loss_per_epoch
        }
        if save_best:
            torch.save(chkpt, chkpt_name)
            print('Best Checkpoint Saved')
        if epoch == 2000:
            snapshot_name = os.path.splitext(chkpt_name)[0] + "_2000.pth"
            torch.save(chkpt, snapshot_name)
            print('Epoch 2000 Checkpoint Saved')
        del chkpt
        
print('DDPM Training Complete.')