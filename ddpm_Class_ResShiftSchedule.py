import math
import os

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image


class Diffusion:
    """ResShift diffusion from the clean residual toward zero."""

    def __init__(self, device, noise_steps=15, img_size=512, kappa=2.0, p=0.3, eta_end=0.999):
        if noise_steps < 2 or kappa <= 0 or p <= 0 or not 0 < eta_end < 1:
            raise ValueError("Invalid ResShift schedule parameters")

        self.device = device
        self.noise_steps = noise_steps
        self.img_size = img_size
        self.kappa = kappa
        self.p = p
        self.eta_end = eta_end

        eta_start = min((0.04 / kappa) ** 2, 0.001)
        if eta_start >= eta_end:
            raise ValueError("eta_1 must be smaller than eta_T")

        # Index 0 is eta_0 = 0; model timesteps are 1,...,T.
        u = torch.linspace(0, 1, noise_steps, device=device)
        eta = torch.exp(
            math.log(eta_start) + u.pow(p) * math.log(eta_end / eta_start)
        )
        self.eta = torch.cat([torch.zeros(1, device=device), eta])
        self.delta = self.eta[1:] - self.eta[:-1]

    def sample_timesteps(self, n):
        return torch.randint(1, self.noise_steps + 1, (n,), device=self.device)

    def noise_images(self, x0, estimate, t):
        """Sample q(x_t | x_0, estimate) directly."""
        eta_t = self.eta[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x0)
        x_t = x0 + eta_t * (estimate - x0) + self.kappa * torch.sqrt(eta_t) * noise
        return x_t, noise

    def sample(self, model, test_dir=None, transform=None):
        """Sample a restored full image for each initial estimate in test_dir."""
        model.eval()
        output_dir = "Generated_Samples/ResShift"
        os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            for imgname in sorted(os.listdir(test_dir)):
                if not imgname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                with Image.open(os.path.join(test_dir, imgname)) as image:
                    estimate = transform(image.convert("RGB")).unsqueeze(0).to(self.device)

                residual_t = self.kappa * torch.randn_like(estimate)
                for step in range(self.noise_steps, 0, -1):
                    t = torch.full((1,), step, dtype=torch.long, device=self.device)
                    residual_0_hat = model(torch.cat([residual_t, estimate], dim=1), t)

                    eta_t = self.eta[step]
                    eta_prev = self.eta[step - 1]
                    delta_t = self.delta[step - 1]
                    mean = (eta_prev / eta_t) * residual_t + (delta_t / eta_t) * residual_0_hat

                    if step > 1:
                        variance = self.kappa**2 * (eta_prev / eta_t) * delta_t
                        residual_t = mean + torch.sqrt(variance) * torch.randn_like(residual_t)
                    else:
                        residual_t = mean

                restored = estimate + residual_t
                result = ((restored.clamp(-1, 1) + 1) / 2).squeeze(0).cpu()
                to_pil_image(result).save(os.path.join(output_dir, imgname))
