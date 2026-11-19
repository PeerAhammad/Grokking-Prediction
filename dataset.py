import numpy as np

from config import (
    MODULUS,
    TRAIN_FRACTION,
    SPLIT_SEED
)


def generate_dataset():
    """
    Generate all modular addition examples.

    Total examples = 97 * 97 = 9409
    """

    a, b = np.meshgrid(
        np.arange(MODULUS),
        np.arange(MODULUS),
        indexing="ij"
    )

    pairs = np.stack(
        [a.ravel(), b.ravel()],
        axis=1
    )

    labels = (pairs[:, 0] + pairs[:, 1]) % MODULUS

    return pairs, labels


def make_orbit_split(
    pairs,
    train_frac=TRAIN_FRACTION,
    split_seed=SPLIT_SEED
):
    """
    Split data by unordered-pair orbits.

    (a,b) and (b,a) always belong to the same split.
    """

    rng = np.random.RandomState(split_seed)

    orbits = []

    for a in range(MODULUS):
        for b in range(a, MODULUS):

            first = a * MODULUS + b
            second = b * MODULUS + a

            if a == b:
                orbits.append([first])
            else:
                orbits.append([first, second])

    # Shuffle complete orbits
    order = rng.permutation(len(orbits))

    n_train = int(train_frac * len(orbits))

    train_orbits = [
        orbits[i] for i in order[:n_train]
    ]

    test_orbits = [
        orbits[i] for i in order[n_train:]
    ]

    # Convert orbits into example indices
    train_idx = np.array(
        [idx for group in train_orbits for idx in group],
        dtype=np.int64
    )

    test_idx = np.array(
        [idx for group in test_orbits for idx in group],
        dtype=np.int64
    )

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return train_idx, test_idx


def check_leakage(train_idx, test_idx):
    """
    Check whether reversed pairs cross between splits.
    """

    train_set = set(train_idx.tolist())
    test_set = set(test_idx.tolist())

    leaked = []

    for idx in test_idx:
        a = idx // MODULUS
        b = idx % MODULUS

        reverse_idx = b * MODULUS + a

        if reverse_idx in train_set:
            leaked.append(idx)

    return leaked


def prepare_dataset():
    """
    Generate, split and validate the dataset.
    """

    pairs, labels = generate_dataset()

    train_idx, test_idx = make_orbit_split(
        pairs
    )

    leaked = check_leakage(
        train_idx,
        test_idx
    )

    # Basic validation
    assert len(train_idx) + len(test_idx) == len(pairs)
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(np.unique(np.concatenate([train_idx, test_idx]))) == len(pairs)
    assert len(leaked) == 0

    return pairs, labels, train_idx, test_idx


if __name__ == "__main__":

    pairs, labels, train_idx, test_idx = prepare_dataset()

    print("\n========== DATASET REPORT ==========")

    print("Total examples:", len(pairs))
    print("Total orbits:", MODULUS * (MODULUS + 1) // 2)

    print("Training examples:", len(train_idx))
    print("Testing examples:", len(test_idx))

    print(
        "Training percentage:",
        round(len(train_idx) / len(pairs) * 100, 2),
        "%"
    )

    print(
        "Testing percentage:",
        round(len(test_idx) / len(pairs) * 100, 2),
        "%"
    )

    print("Leaked examples:", len(check_leakage(train_idx, test_idx)))

    print("\nSample examples:")

    for i in range(5):
        a, b = pairs[i]
        print(f"{a} + {b} mod {MODULUS} = {labels[i]}")

    print("\nSUCCESS: Dataset validation passed!")