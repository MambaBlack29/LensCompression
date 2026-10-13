import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchmetrics.functional.image import structural_similarity_index_measure as ssim, peak_signal_noise_ratio as psnr
from tqdm import tqdm

class GeneratedDataset(Dataset):
    def __init__(self, dir_gt, dir_generated, dir_sample):
        self.pairs, self.transform = [], ToTensor()
        for folder in sorted(os.listdir(dir_generated)):
            num = folder.removeprefix("output")
            if not folder.startswith("output") or not num.isdigit() or int(num) < 501:
                continue
            generated = os.path.join(dir_generated, folder, f"0{folder}.png")
            if not os.path.isfile(generated):
                continue
            gt = os.path.join(dir_gt, f"img{num}.jpg")
            sample = os.path.join(dir_sample, f"output{num}.png")
            for path in (gt, sample):
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Missing image: {path}")
            self.pairs.append((generated, sample, gt))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        generated_path, sample_path, gt_path = self.pairs[index]
        with Image.open(generated_path) as img:
            generated = self.transform(img.convert("RGB"))
        with Image.open(sample_path) as img:
            sample = self.transform(img.convert("RGB"))
        with Image.open(gt_path) as img:
            gt = self.transform(img.convert("RGB"))[:, :, 512:]
        return generated, sample, gt

dir_gt = "Dataset/f2fGT"
# dir_generated = "outputs/Generated_Samples_3_1524/Images"
dir_generated = "Generated_Samples/Images"
dir_sample = "Dataset/testestimatesW2L"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = GeneratedDataset(dir_gt, dir_generated, dir_sample)
if not len(dataset):
    raise ValueError("No generated images numbered 501 onwards found.")
dataloader = DataLoader(dataset, batch_size=12, shuffle=False)
scores = {"sample": [0.0, 0.0], "gt": [0.0, 0.0]}
sample_gt_score = [0.0, 0.0]
count = 0

with torch.no_grad():
    for generated, sample, gt in tqdm(dataloader):
        generated, sample, gt = generated.to(device), sample.to(device), gt.to(device)
        for name, target in (("sample", sample), ("gt", gt)):
            if generated.shape != target.shape:
                raise ValueError(f"{name} shape mismatch: {generated.shape} vs {target.shape}")
            scores[name][0] += ssim(generated, target, data_range=1.0, reduction="sum").item()
            scores[name][1] += psnr(generated, target, data_range=1.0, dim=(1, 2, 3), reduction="sum").item()

        sample_gt_score[0] += ssim(sample, gt, data_range=1.0, reduction="sum").item()
        sample_gt_score[1] += psnr(sample, gt, data_range=1.0, dim=(1, 2, 3), reduction="sum").item()
        count += generated.shape[0]

print(f"Images evaluated: {count}")
for name, (ssim_sum, psnr_sum) in scores.items():
    print(f"(pred, {name}) Mean SSIM: {ssim_sum / count:.6f}, Mean PSNR: {psnr_sum / count:.4f} dB")

print(f"(sample, gt) Mean SSIM: {sample_gt_score[0] / count:.6f}, Mean PSNR: {sample_gt_score[1] / count:.4f} dB")
