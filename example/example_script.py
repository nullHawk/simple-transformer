import torch
from simple_transformer import Transformer

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example input data
x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

# Model parameters
src_pad_idx = 0
trg_pad_idx = 0
src_vocab_size = 10
trg_vocab_size = 10

# Initialize and run the model
model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
out = model(x, trg[:, :-1])

# Output shape
print(out.shape)
