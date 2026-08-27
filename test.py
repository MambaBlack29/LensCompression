import math
import time
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset import Wide2LongDatasetAfterInitialEstimate
from ddpm_Class_LinearSchedule import Diffusion
from UNET import UNet


NUM_ITERATIONS = 20
IMG_SIZE = 512
BATCH_SIZE = 1  # Increase only if it fits in your GPU memory.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = Path(__file__).resolve().parent / "Dataset"

transform_concatdataset = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, 2 * IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
transform_estimate = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
transform_aug = transforms.Compose(
    [
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
)

dataset = Wide2LongDatasetAfterInitialEstimate(
    dir_concatdataset=str(data_dir / "f2fGT"),
    dir_estimate=str(data_dir / "trainestimatesW2L"),
    transform_concatdataset=transform_concatdataset,
    transform_estimate=transform_estimate,
    transform_aug=transform_aug,
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = UNet(
    device=device,
    c_in=6,
    c_out=3,
    time_dim=256,
    upattn=[False, False, False],
    downattn=[False, False, False],
    img_size=IMG_SIZE,
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
diffusion = Diffusion(
    noise_steps=1000,
    img_size=IMG_SIZE,
    device=device,
    beta_start=1e-4,
    beta_end=5e-3,
)
criterion = torch.nn.L1Loss(reduction="none")
fidelity = torch.nn.MSELoss(reduction="none")

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Device: {device}")
print(f"Trainable parameters: {num_params:,}")
print(f"Dataset pairs: {len(dataset):,}")
print(f"Batch size: {BATCH_SIZE}")

model.train()
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

step_times = []
data_iterator = iter(dataloader)
for iteration in range(NUM_ITERATIONS):
    iteration_start = time.perf_counter()
    estimate, ground_truth = next(data_iterator)
    estimate = estimate.float().to(device)
    ground_truth = ground_truth.float().to(device)

    residual = ground_truth - estimate
    timesteps = diffusion.sample_timesteps(estimate.shape[0]).to(device)
    noisy_residual, noise = diffusion.noise_images(residual, timesteps)
    predicted_noise = model(
        torch.cat([noisy_residual, estimate], dim=1), timesteps
    )

    mse = criterion(noise, predicted_noise)
    alpha_hat = diffusion.return_alpha_hat()[timesteps]
    snr = alpha_hat / (1 - alpha_hat)
    alpha_hat = alpha_hat.view(-1, 1, 1, 1)
    reconstructed = (
        (noisy_residual - torch.sqrt(1 - alpha_hat) * predicted_noise)
        / torch.sqrt(alpha_hat)
    ) + estimate
    weights = (1 / (snr + 1) ** 0.5)[:, None, None, None]
    loss = (weights * mse).mean()
    loss = loss + 0.01 * fidelity(reconstructed, ground_truth).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    step_times.append(time.perf_counter() - iteration_start)
    print(
        f"Iteration {iteration + 1:02d}/{NUM_ITERATIONS}: "
        f"{step_times[-1]:.2f}s, loss={loss.item():.5f}"
    )

average_step = sum(step_times) / len(step_times)
steady_state_step = (
    sum(step_times[1:]) / len(step_times[1:]) if len(step_times) > 1 else average_step
)
steps_per_epoch = math.ceil(len(dataset) / BATCH_SIZE)
print(f"\nAverage step time: {average_step:.2f}s")
print(f"Average excluding first step: {steady_state_step:.2f}s")
print(f"Estimated time per epoch: {steady_state_step * steps_per_epoch / 60:.1f} minutes")
if device.type == "cuda":
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"Peak GPU memory: {peak_memory:.2f} GiB")