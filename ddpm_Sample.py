import os
import PIL
import torch
from UNET import UNet
from ddpm_Class_SigmoidSchedule import Diffusion
# from ddpm_Class_LinearSchedule import Diffusion
# from ddpm_Class_SigmoidSchedule_noclamping import Diffusion
import torchvision.transforms as transforms

#Checking for GPU support
if torch.cuda.is_available():
    print("Running on GPU.\nGPU name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())
    device = torch.device('cuda:0')
    print("Device:",device)
    
else:
    print("GPU not available!!!")

        
# %%   
#Defining the Unet noise predictor
in_channels = 6
img_size = 512

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_dir = 'Dataset/testestimatesW2L'

upattn = [False, False, False]
downattn = [False, False, False]
model = UNet(device = device, img_size = img_size, time_dim = 256, downattn = downattn, upattn = upattn, c_in = 6, c_out = 3).to(device)

checkpoint = torch.load('ddpm_T1000_512x512_Linear_L10point01FidelityLoss.pth')
print("Loading: ", 'ddpm_T1000_512x512_Linear_L10point01FidelityLoss.pth')

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

diffusion = Diffusion(device = device, noise_steps = 1000, img_size = img_size, beta_start = 1e-4, beta_end = 5e-3, a = 8)
# diffusion = Diffusion(device = device, noise_steps = 1000, img_size = img_size, beta_start = 1e-4, beta_end = 5e-3)
diffusion.sample(model = model, test_dir = test_dir, transform = transform)