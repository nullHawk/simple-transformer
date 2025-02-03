
# Simple Transformer

A simple and modular implementation of the Transformer model in PyTorch. This package provides an intuitive interface to build Transformer models for a variety of NLP tasks like translation, summarization, and more.

---

## 🌟 Features

- **Encoder**, **Decoder**, and **Self-Attention** layers implemented.
- Highly **customizable** architecture (e.g., number of layers, attention heads, embedding size).
- Supports **Masked** and **Unmasked** attention mechanisms.
- Easy to **integrate** and **extend** for different use cases.

---

## 🚀 Installation

### Via pip

You can install the package directly from PyPI:

```bash
pip install simple-transformer
```

### From Source

Alternatively, clone the repository and install it locally:

```bash
git clone https://github.com/nullHawk/simple-transformer.git
cd simple-transformer
pip install .
```

---

## 📝 Usage

### Basic Transformer Model

Here's how you can use the basic Transformer model in your project:

```python
import torch
from simple_transformer import Transformer

# Define input parameters
src_vocab_size = 10
trg_vocab_size = 10
src_pad_idx = 0
trg_pad_idx = 0

# Initialize the Transformer model
model = Transformer(
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx,
)

# Example input (batch_size=2, sequence_length=9)
src = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]])
trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]])

# Forward pass through the Transformer model
out = model(src, trg[:, :-1])
print(out.shape)
```

---

## 🔧 Components

- **Encoder**: Processes the source sequence, extracting context information.
- **Decoder**: Generates the target sequence based on the encoded input.
- **Self-Attention**: Enables each word in a sequence to focus on other words in the same sequence.
- **Transformer**: The central model combining the encoder and decoder for sequence-to-sequence tasks.

---

## 📦 Requirements

- Python >= 3.7
- PyTorch >= 1.9.0

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## 🙏 Acknowledgements

- This package is inspired by the [original Transformer model](https://arxiv.org/abs/1706.03762).

---

By adjusting the structure and including headers for each section, the README is now cleaner and more visually appealing. This layout should enhance readability and user experience when browsing through your GitHub repository!