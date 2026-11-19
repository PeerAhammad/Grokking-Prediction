import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

data = pd.read_csv(ROOT / "results" / "seed_0.csv")

# Accuracy graph
plt.figure(figsize=(10, 5))

plt.plot(data["step"], data["train_accuracy"], label="Training Accuracy")
plt.plot(data["step"], data["test_accuracy"], label="Testing Accuracy")

plt.xlabel("Training Steps")
plt.ylabel("Accuracy")
plt.title("Grokking Experiment - Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(ROOT / "results" / "seed_0_accuracy.png")
plt.show()

# Loss graph
# Testing accuracy - zoomed view
plt.figure(figsize=(10, 5))

plt.plot(
    data["step"],
    data["test_accuracy"],
    label="Testing Accuracy"
)

plt.xlabel("Training Steps")
plt.ylabel("Testing Accuracy")
plt.title("Testing Accuracy - Zoomed View")

plt.ylim(0, 0.02)

plt.legend()
plt.grid(True)

plt.savefig(ROOT / "results" / "seed_0_test_accuracy_zoomed.png")
plt.show()