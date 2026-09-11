# %%
import torch
import os
import torchvision.transforms as transforms
from UNET import UNet
from ddpm_Class_SigmoidSchedule import Diffusion as diffusion_sigmoidbeta

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("Running on GPU:", torch.cuda.get_device_name(0))
else:
    device = 'cpu'
    print("Running on CPU.")

# %%
img_size = 512
dir_val = 'Dataset/valestimatesW2L'
chkpt_name = 'ddpm_T1000_512x512_Linear_L1_FL_0p01.pth'

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

diffusion_sample = diffusion_sigmoidbeta(
    noise_steps=1000,
    img_size=img_size,
    device=device,
    beta_start=1e-4,
    beta_end=5e-3,
    a=8
)

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