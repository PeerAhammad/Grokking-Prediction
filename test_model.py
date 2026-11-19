import torch

from models.transformer import GrokkingTransformer

# Create the model
model = GrokkingTransformer()

# Sample input: 12 + 15 =
x = torch.tensor([[12, 15, 97]])

# Run the model
output = model(x)

print("Input:", x)
print("Output shape:", output.shape)
print("Predicted answer:", output.argmax(dim=1).item())
print("Number of parameters:", sum(p.numel() for p in model.parameters()))

# Check output
assert output.shape == (1, 97)

print("\nSUCCESS: Transformer model is working!")