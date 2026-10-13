# %%
import re
import pandas as pd
import matplotlib.pyplot as plt

log = "logs/train_6_1.log"
with open(log, "r") as f:
    text = f.read()

num = r"(?:[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|inf)"

rows = []

for m in re.finditer(r"Epoch No:\s*(\d+)(.*?)(?=Epoch No:|\Z)", text, re.DOTALL):
    epoch = int(m.group(1))
    block = m.group(2)

    x = re.search(
        # rf"Non-FL Loss:\s*({num}).*?"
        # rf"Fidelity Loss:\s*({num}).*?"
        rf"Loss:\s*({num}).*?"
        rf"Lowest loss:\s*({num})",
        block, re.DOTALL
    )

    if x:
        rows.append([
            epoch,
            float(x.group(1)),
            float(x.group(2)),
            # float(x.group(3)),
            # float(x.group(4))
        ])

df = pd.DataFrame(rows, columns=[
    "Epoch", "Loss", "Lowest Loss"
])

print(df)
df.to_csv("training_losses.csv", index=False)

# %%
# plt.plot(df["Epoch"], df["Non-FL Loss"], label="Non-FL Loss")
# plt.plot(df["Epoch"], df["Fidelity Loss"], label="Fidelity Loss")
r = slice(0, len(rows))
plt.plot(df["Epoch"][r], df["Loss"][r], label="Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("training_losses.png", dpi=300, bbox_inches="tight")
plt.close()
