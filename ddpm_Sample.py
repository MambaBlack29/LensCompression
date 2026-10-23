# %%
import torch
import os
import torchvision.transforms as transforms
from UNET import UNet
from ddpm_Class_CosineSchedule import Diffusion as diffusion_cosbeta
from ddpm_Class_SigmoidSchedule import Diffusion as diffusion_sigmoidbeta
from ddpm_Class_LinearSchedule import Diffusion as diffusion_linearbeta
from ddpm_Class_ResShiftSchedule import Diffusion as diffusion_resshift

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("Running on GPU:", torch.cuda.get_device_name(0))
else:
    device = 'cpu'
    print("Running on CPU.")

# %%
img_size = 512
dir_val = 'Dataset/testestimatesW2L'
method = 'resshift'  # 'ddpm' keeps the existing sampling path
chkpt_name = ('checkpoints/resshift_residual_T30_512x512_L2.pth' if method == 'resshift'
              else 'checkpoints/ddpm_T1000_512x512_Linear_L2_ssim_1_2000.pth')

upattn = [False, False, False]
downattn = [False, False, False]

transform_estimate = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

model = UNet(
    device=device,
    c_in=6,
    c_out=3,
    time_dim=256,
    upattn=upattn,
    downattn=downattn,
    img_size=img_size
).to(device)

if not os.path.exists(chkpt_name):
    raise FileNotFoundError(f"Checkpoint not found: {chkpt_name}")

print("Loading Checkpoint:", chkpt_name)
chkpt = torch.load(chkpt_name, map_location=device)
model.load_state_dict(chkpt['model_state_dict'])

if method == 'resshift':
    diffusion_sample = diffusion_resshift(noise_steps=15, img_size=img_size, device=device,
                                          kappa=2.0, p=0.3, eta_end=0.999)
else:
    # diffusion_sample = diffusion_cosbeta(
    #     noise_steps=1000,
    #     img_size=img_size,
    #     device=device,
    #     s=0.008
    # )
    diffusion_sample = diffusion_sigmoidbeta(
        noise_steps=1000,
        img_size=img_size,
        device=device,
        beta_start=1e-4,
        beta_end=5e-3,
        a=8
    )
    # diffusion_sample = diffusion_linearbeta(
    #     noise_steps=1000,
    #     img_size=img_size,
    #     device=device,
    #     beta_start=1e-4,
    #     beta_end=5e-3
    # )

# %%
print("Starting Evaluation")
model.eval()

with torch.no_grad():
    diffusion_sample.sample(
        model=model,
        test_dir=dir_val,
        transform=transform_estimate
    )

print("Evaluation Complete")
