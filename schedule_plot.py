import torch
import matplotlib.pyplot as plt

noise_steps = 1000
s = 0.008
beta_start, beta_end, a = 1e-4, 8e-3, 8
gamma = 0.5
# beta_start = 1e-6
# beta_end = 1e-2

# Cosine schedule
steps = torch.linspace(0, noise_steps, noise_steps + 1)
f = torch.cos(((steps / noise_steps + s) / (1 + s)) * torch.pi / 2)**2
alpha_hat = f / f[0]
beta_cos = (1 - alpha_hat[1:] / alpha_hat[:-1]).clamp(max=0.999)
beta_cosine = ((1 - alpha_hat[:-1]) / (1 - alpha_hat[1:]) * beta_cos).clamp(max=0.999)
# alpha_hat_cosine = alpha_hat[1:]
alpha_hat_cosine = torch.cumprod(1 - beta_cosine, dim=0)
snr_cosine = alpha_hat_cosine / (1 - alpha_hat_cosine + 1e-5)
baseline_cosine = (1 - beta_cosine)*(1-alpha_hat_cosine)/beta_cosine
weight_cosine =  baseline_cosine/((1 + snr_cosine)**gamma)

# Sigmoid schedule
steps = torch.linspace(-6, 6, noise_steps)
beta_sigmoid = torch.sigmoid(a * steps) * (beta_end - beta_start) + beta_start
alpha_hat_sigmoid = torch.cumprod(1 - beta_sigmoid, dim=0)
snr_sigmoid = alpha_hat_sigmoid / (1 - alpha_hat_sigmoid + 1e-5)
baseline_sigmoid = (1 - beta_sigmoid)*(1-alpha_hat_sigmoid)/beta_sigmoid
weight_sigmoid =  baseline_sigmoid/((1 + snr_sigmoid)**gamma)

# Linear schedule
beta_linear = torch.linspace(start = beta_start, end = beta_end, steps = noise_steps)
alpha_hat_linear = torch.cumprod(1-beta_linear, dim = 0)
snr_linear = alpha_hat_linear / (1 - alpha_hat_linear + 1e-5)
baseline_linear = (1 - beta_linear)*(1-alpha_hat_linear)/beta_linear
weight_linear =  baseline_linear/((1 + snr_linear)**gamma)

for name, beta, alpha_hat in [
    ("Cosine", beta_cosine, alpha_hat_cosine),
    ("Sigmoid", beta_sigmoid, alpha_hat_sigmoid),
    ("Linear", beta_linear, alpha_hat_linear)
]:
    print(f"\n{name}")
    print(f"beta:      initial = {beta[0].item():.10e}, final = {beta[-1].item():.10e}")
    print(f"alpha_hat: initial = {alpha_hat[0].item():.10e}, final = {alpha_hat[-1].item():.10e}")

t = torch.arange(1, noise_steps + 1) / noise_steps
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# for name, beta, alpha_hat, color in [
#     # ("Cosine", beta_cosine, alpha_hat_cosine, "tab:blue"),
#     ("Sigmoid", beta_sigmoid, alpha_hat_sigmoid, "tab:orange"),
#     ("Linear", beta_linear, alpha_hat_linear, "tab:green")
# ]:
#     axes[0].plot(t, alpha_hat, color=color, label=name)
#     axes[1].plot(t, beta, color=color, label=name)

# axes[0].set_title(r"$\hat{\alpha}_t$")
# axes[0].set_ylabel(r"$\hat{\alpha}_t$")
# axes[1].set_title(r"$\beta_t$")
# axes[1].set_ylabel(r"$\beta_t$")

for name, snr, weight, color in [
    ("Cosine", snr_cosine, weight_cosine, "tab:blue"),
    ("Sigmoid", snr_sigmoid, weight_sigmoid, "tab:orange"),
    ("Linear", snr_linear, weight_linear, "tab:green")
]:
    log_snr = torch.log(snr)
    axes[0].plot(t, weight, color=color, label=name)
    axes[1].plot(t, log_snr, color=color, label=name)

axes[0].set_title(r"$weight_t$ vs $\log(SNR_t)$")
axes[0].set_xlabel(r"$\log(SNR_t)$")
axes[0].set_ylabel(r"$weight_t$")
axes[1].set_title(r"$\log(SNR_t)$ vs timestep")
axes[1].set_xlabel(r"$t$")
axes[1].set_ylabel(r"$\log(SNR_t)$")

for ax in axes:
    # ax.set_xlabel(r"$t/T$")
    # ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()

fig.tight_layout()
fig.savefig("schedules.png", dpi=300, bbox_inches="tight")
plt.show()