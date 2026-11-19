# ==========================================
# Grokking Research - Configuration
# ==========================================

# Dataset settings
MODULUS = 97
VOCAB_SIZE = MODULUS + 1
EQUALS_TOKEN = MODULUS

TRAIN_FRACTION = 0.30
SPLIT_SEED = 10000

# Transformer settings
EMBED_DIM = 64
NUM_HEADS = 4
MLP_DIM = 128
NUM_BLOCKS = 1
MAX_SEQUENCE_LENGTH = 3

# Training settings
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1.0
BATCH_SIZE = 512
TOTAL_STEPS = 20000
EVAL_EVERY = 100
PRINT_EVERY = 1000

# Experimental settings
RANDOM_SEEDS = list(range(30))

# Grokking measurement
TRAIN_ACCURACY_THRESHOLD = 0.95
TEST_ACCURACY_THRESHOLD = 0.80
CONSECUTIVE_EVALS = 5