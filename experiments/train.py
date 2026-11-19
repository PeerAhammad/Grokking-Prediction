import sys
import csv
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import *
from dataset import prepare_dataset
from models.transformer import GrokkingTransformer


def train(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nStarting seed {seed} on {device}")

    # Prepare dataset
    pairs, labels, train_idx, test_idx = prepare_dataset()

    X = np.concatenate(
        [pairs, np.full((len(pairs), 1), EQUALS_TOKEN)],
        axis=1
    )

    X = torch.tensor(X, dtype=torch.long, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)

    train_idx = torch.tensor(train_idx, device=device)
    test_idx = torch.tensor(test_idx, device=device)

    # Initialize model
    model = GrokkingTransformer().to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    # Tracking variables
    train_streak = 0
    test_streak = 0

    memorization_step = None
    generalization_step = None

    # Prepare CSV file
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)

    csv_path = results_dir / f"seed_{seed}.csv"

    with open(csv_path, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "step",
            "loss",
            "train_accuracy",
            "test_accuracy"
        ])

        # Training loop
        for step in range(1, TOTAL_STEPS + 1):

            model.train()

            batch_indices = train_idx[
                torch.randint(
                    0,
                    len(train_idx),
                    (BATCH_SIZE,),
                    device=device
                )
            ]

            predictions = model(X[batch_indices])

            loss = criterion(
                predictions,
                y[batch_indices]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate periodically
            if step % EVAL_EVERY == 0:

                model.eval()

                with torch.no_grad():

                    train_predictions = model(X[train_idx]).argmax(dim=1)
                    test_predictions = model(X[test_idx]).argmax(dim=1)

                    train_accuracy = (
                        train_predictions == y[train_idx]
                    ).float().mean().item()

                    test_accuracy = (
                        test_predictions == y[test_idx]
                    ).float().mean().item()

                # Save measurements
                writer.writerow([
                    step,
                    loss.item(),
                    train_accuracy,
                    test_accuracy
                ])

                # Check memorization threshold
                if train_accuracy >= TRAIN_ACCURACY_THRESHOLD:
                    train_streak += 1
                else:
                    train_streak = 0

                # Check generalization threshold
                if test_accuracy >= TEST_ACCURACY_THRESHOLD:
                    test_streak += 1
                else:
                    test_streak = 0

                if (
                    memorization_step is None
                    and train_streak >= CONSECUTIVE_EVALS
                ):
                    memorization_step = step

                if (
                    generalization_step is None
                    and test_streak >= CONSECUTIVE_EVALS
                ):
                    generalization_step = step

                # Display progress every 1000 steps
                if step % PRINT_EVERY == 0:

                    print(
                        f"Step {step:5d} | "
                        f"Loss {loss.item():.4f} | "
                        f"Train {train_accuracy:.3f} | "
                        f"Test {test_accuracy:.3f}"
                    )


    # Prediction diagnostics
    model.eval()

    with torch.no_grad():

        test_logits = model(X[test_idx])
        test_predictions = test_logits.argmax(dim=1)

    correct = (test_predictions == y[test_idx]).sum().item()

    prediction_counts = torch.bincount(
        test_predictions,
        minlength=MODULUS
    )

    target_counts = torch.bincount(
        y[test_idx],
        minlength=MODULUS
    )

    print("\n========== PREDICTION DIAGNOSTICS ==========")

    print("Correct predictions:", correct)
    print("Total test examples:", len(test_idx))
    print("Recalculated accuracy:", correct / len(test_idx))

    print("Unique predicted classes:",
      (prediction_counts > 0).sum().item())

    print("Unique actual classes:",
      (target_counts > 0).sum().item())

    print("Most predicted classes:",
      torch.topk(prediction_counts, 10).indices.tolist())

    print(
    "Most predicted class frequencies:",
    torch.topk(prediction_counts, 10).values.tolist()
    )

    print(
    "Most common actual class frequency:",
    target_counts.max().item()
    )

    print(
    "Majority-class baseline:",
    target_counts.max().item() / len(test_idx)
    )

    print(
    "Uniform random baseline:",
    1 / MODULUS
    )

    # Save final model
    model_path = results_dir / f"model_seed_{seed}.pt"
    torch.save(model.state_dict(), model_path)

    print("Model saved:", model_path)
    
    
    # Final report
    print("\n========== EXPERIMENT REPORT ==========")

    print("Seed:", seed)
    print("Memorization step:", memorization_step)
    print("Generalization step:", generalization_step)

    if generalization_step is None:
        print("Status: Generalization threshold not reached.")
    else:
        print("Status: Generalization threshold reached.")

    print("Results saved to:", csv_path)


if __name__ == "__main__":

    train(seed=0)