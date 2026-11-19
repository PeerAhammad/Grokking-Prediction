# Predicting Grokking in Transformers Using Initialization-Time Spectral Features

> **Research Status:** Experimental Foundation Completed · Preliminary Experiments in Progress

## 1. Overview

Grokking is a phenomenon in neural networks where a model initially memorizes its training data but only begins to generalize to unseen examples after extended training.

This research investigates whether **initialization-time spectral features of Transformer weight matrices can help predict the emergence of grokking**.

The central research question is:

> Can the spectral properties of a Transformer at initialization predict whether, and potentially when, it will transition from memorization to generalization?

The project uses modular addition as a controlled learning task and a Transformer implemented in PyTorch.

**Note:** The hypothesis has not yet been experimentally established. The current work focuses on building and validating the experimental framework.

---

## 2. Research Objectives

- Implement a reproducible Transformer training framework.
- Study grokking using modular addition.
- Prevent data leakage through orbit-based dataset splitting.
- Extract spectral features from model weights before training.
- Investigate relationships between initialization-time features and subsequent generalization.
- Evaluate prediction performance across multiple random seeds.

---

## 3. Experimental Task

We use modular addition:

\[
(a+b)\bmod 97
\]

For example:

\[
(12+15)\bmod 97=27
\]

This task is useful because it provides a small, controlled mathematical environment in which memorization and generalization can be measured separately.

### Dataset Configuration

| Property | Value |
|---|---:|
| Modulus | 97 |
| Total examples | 9,409 |
| Total orbits | 4,753 |
| Training examples | 2,821 |
| Testing examples | 6,588 |
| Training split | ~30% |
| Testing split | ~70% |
| Data leakage | 0 |

### Orbit-Based Splitting

Modular addition is commutative:

\[
(a+b)\bmod 97=(b+a)\bmod 97
\]

A conventional random split could place `(a,b)` in training and `(b,a)` in testing, creating information leakage.

To prevent this, we group both ordered pairs into the same orbit before splitting the dataset.

**Validation result:** 0 leaked examples. Dataset validation passed.

---

## 4. Model Architecture

A small Transformer is used to study grokking under controlled conditions.

| Component | Configuration |
|---|---|
| Framework | PyTorch |
| Embedding dimension | 64 |
| Attention heads | 4 |
| Transformer blocks | 1 |
| MLP dimension | 128 |
| Activation | GELU |
| Normalization | LayerNorm |
| Parameters | 46,369 |
| Input | `[a, b, equals]` |
| Prediction | Modular addition result |

### Training Configuration

| Parameter | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 1.0 |
| Batch size | 512 |
| Total training steps | 20,000 |
| Evaluation interval | 100 steps |
| Progress logging interval | 1,000 steps |
| Planned random seeds | 30 |

---

## 5. Research Methodology

The planned experimental pipeline is:

```text
Modular Addition Dataset
          |
          v
Orbit-Based Data Splitting
          |
          v
Transformer Initialization
          |
          +-------------------------+
          |                         |
          v                         v
  Spectral Feature            Model Training
    Extraction                     |
          |                         v
          |                 Train/Test Evaluation
          |                         |
          +------------+------------+
                       |
                       v
              Prediction Analysis
                       |
                       v
              Statistical Evaluation
```

### Spectral Analysis

The main research component is the extraction of spectral properties from Transformer weight matrices before training.

Potential features include:

- Largest singular value.
- Spectral norm.
- Singular value distribution.
- Stable rank.
- Effective rank.
- Spectral concentration.

These features will be investigated to determine whether they contain predictive information about later training behaviour.

The feature set and prediction method are still to be implemented and evaluated.

---

## 6. Preliminary Experimental Results

An initial experiment was completed using random seed 0.

### Results

| Metric | Result |
|---|---:|
| Training steps | 20,000 |
| Training accuracy | ~100% |
| Testing accuracy | 0.243% |
| Uniform random reference | 1.03% |
| Memorization | Observed |
| Generalization threshold reached | No |

### Interpretation

The model successfully memorized the training examples but failed to generalize to the unseen test examples during the initial 20,000 training steps.

This experiment does not demonstrate a successful grokking transition.

The training configuration and implementation must be investigated before extending the experiments to multiple seeds.

---

## 7. Current Project Progress

| Component | Status |
|---|---|
| Research question and hypothesis | Completed |
| Project structure | Completed |
| Dataset generation | Completed |
| Orbit-based splitting | Completed |
| Data leakage validation | Passed |
| Transformer implementation | Completed |
| Model execution test | Passed |
| GPU training pipeline | Completed |
| Initial experiment | Completed |
| Accuracy and loss tracking | Completed |
| Result visualization | Completed |
| GitHub repository setup | Completed |
| Training pipeline investigation | Pending |
| Spectral feature extraction | Pending |
| Multiple-seed experiments | Pending |
| Prediction model | Pending |
| Final evaluation | Pending |

---

## 8. Technology Stack

- **Language:** Python 3.10
- **Deep Learning:** PyTorch
- **Numerical Computing:** NumPy
- **Data Analysis:** Pandas
- **Visualization:** Matplotlib
- **Development Environment:** Visual Studio Code
- **Hardware:** NVIDIA GPU with CUDA support

---

## 9. Project Structure

```text
Grokking-Prediction/
│
├── analysis/
│   ├── evaluate.py
│   └── spectral_analysis.py
│
├── data/
│
├── experiments/
│   └── train.py
│
├── models/
│   └── transformer.py
│
├── results/
│   ├── seed_0.csv
│   └── model_seed_0.pt
│
├── config.py
├── dataset.py
├── main.py
├── test_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 10. Next Steps

The immediate priority is to establish a reliable experimental setup before conducting large-scale experiments.

1. Investigate the current training configuration and verify the implementation.
2. Establish conditions under which grokking can be observed.
3. Run experiments across multiple random seeds.
4. Extract initialization-time spectral features.
5. Construct a method for predicting generalization behaviour.
6. Evaluate predictions against actual experimental outcomes.
7. Analyse results and document the findings.

---

## 11. Research Scope and Limitations

- Current results are based on one random seed.
- The initial experiment demonstrated memorization without generalization.
- Spectral feature extraction and predictive modelling are not yet complete.
- No predictive relationship has been established.
- Further experiments are necessary before drawing conclusions.

The research will distinguish between observed experimental results and hypotheses that remain to be tested.

---

## 12. Research Summary

This project explores the possibility of predicting grokking in Transformers using spectral information available at initialization.

The experimental foundation has been implemented, including modular addition dataset generation, leakage-resistant orbit splitting, Transformer architecture, GPU training, and preliminary evaluation.

The next phase focuses on validating the training setup and investigating whether initialization-time spectral features can provide useful information about future generalization behaviour.

**Research status: Experimental framework implemented; predictive hypothesis under investigation.**
