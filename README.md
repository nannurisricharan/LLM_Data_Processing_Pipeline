# LLM Data Preprocessing Pipeline

A self-contained Jupyter notebook implementing the complete data preprocessing pipeline
required to prepare raw text for training a GPT-style large language model — from raw
characters all the way to the dense embedding tensors that enter a Transformer.

---

## Pipeline at a Glance

```
Raw Text
   |  Stage 1 - Tokenization
   v
Tokenized Text  (list of string tokens)
   |  Stage 2 - Vocabulary + Token IDs
   v
Token IDs  (integers)
   |  Stage 3 - Token Embeddings  +  DataLoader
   v
Token Embedding Vectors  (B x S x D float tensor)
   |  Stage 4 - Positional Embeddings
   v
Positional Embedding Vectors  (S x D float tensor)
   |  Stage 5 - Combine
   v
Input Embeddings  (B x S x D) -- ready for the Transformer
```

`B` = batch size, `S` = sequence length, `D` = embedding dimension.

---

## Contents

| # | Section | Core components |
|---|---|---|
| 1 | **Tokenization** | `re.split` with a punctuation-aware pattern |
| 2 | **Token IDs** | Vocabulary dict, `TokenizerV1`, `TokenizerV2`, BPE via `tiktoken` |
| 3 | **Token Embeddings + DataLoader** | `GPTDatasetV1`, `create_dataloader_v1`, `nn.Embedding` |
| 4 | **Positional Embeddings** | Learned absolute positional `nn.Embedding` |
| 5 | **Input Embeddings** | Element-wise sum with broadcasting |

---

## File Structure

```
.
├── LLM_Data_Preprocessing_Pipeline.ipynb   # Main notebook
├── the-verdict.txt                          # Sample corpus (required)
└── README.md
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install tiktoken torch

# Run
jupyter notebook LLM_Data_Preprocessing_Pipeline.ipynb
```

> Requires **Python >= 3.9** and **PyTorch >= 2.0**.

---

## Concept Breakdown

### Stage 1 - Raw Text to Tokenized Text

Tokenization splits a continuous string into discrete units called **tokens**.
A regular expression pattern simultaneously handles whitespace and punctuation,
producing one token per word and one per punctuation mark.

Whitespace characters are discarded after splitting. Retaining them would increase
sequence length without adding semantic value for most LLM tasks, though code models
(which depend on indentation) may benefit from keeping them.

---

### Stage 2 - Tokenized Text to Token IDs

A **vocabulary** is a dictionary mapping every unique token to a sequential integer.
Two special tokens are always included:

| Token | Purpose |
|---|---|
| `<\|unk\|>` | Substitute for any word not found in the vocabulary |
| `<\|endoftext\|>` | Boundary marker between independent documents in a batch |

**TokenizerV1** encodes text to IDs and decodes IDs back to text, but raises a
`KeyError` for unseen words — suitable only when the full vocabulary is guaranteed.

**TokenizerV2** adds an OOV fallback, silently mapping any unseen word to `<|unk|>`,
making it safe for open-domain text.

#### Byte Pair Encoding (BPE)

Production GPT models use BPE instead of simple regex tokenisation.
The algorithm works by:

1. Starting with a base vocabulary of individual bytes or characters.
2. Counting every adjacent pair of tokens across the full corpus.
3. Merging the most frequent pair into a single new token.
4. Repeating until the vocabulary reaches its target size.

GPT-2 uses a vocabulary of **50,257** tokens. Because any string can be decomposed
into sub-word byte pieces, BPE never needs a `<|unk|>` token. We load OpenAI's
pre-trained GPT-2 BPE encoding through the `tiktoken` library.

---

### Stage 3 - Token IDs to Token Embeddings (+ DataLoader)

#### Sliding-Window Data Sampling

LLMs are trained on a **next-token prediction** objective: given a window of tokens,
predict the next one. A sliding window of width `max_length` steps through the full
token sequence, creating `(input, target)` pairs where the target is the input shifted
right by one position.

```
tokens  : [t0, t1, t2, t3, t4, t5, ...]
window 1 -> input = [t0, t1, t2, t3]   target = [t1, t2, t3, t4]
window 2 -> input = [t1, t2, t3, t4]   target = [t2, t3, t4, t5]  (stride = 1)
window 2 -> input = [t4, t5, t6, t7]   target = [t5, t6, t7, t8]  (stride = 4)
```

The `stride` controls overlap: `stride = 1` maximises data reuse;
`stride = max_length` creates non-overlapping windows with no redundancy.

#### GPTDatasetV1

A `torch.utils.data.Dataset` that applies the sliding window to a full corpus and
stores all resulting `(input_ids, target_ids)` tensor pairs. Implements the
standard `__len__` and `__getitem__` interface.

| Parameter | Type | Description |
|---|---|---|
| `txt` | `str` | Raw text corpus |
| `tokenizer` | tiktoken encoding | Tokenizer to encode the text |
| `max_length` | `int` | Context window width |
| `stride` | `int` | Step size between consecutive windows |

#### create_dataloader_v1

A factory function wrapping `GPTDatasetV1` in a PyTorch `DataLoader`, which provides:
- **Batching** — groups windows into mini-batches for GPU efficiency
- **Shuffling** — randomises order each epoch to stabilise gradient updates
- **`drop_last=True`** — discards the final incomplete batch to prevent shape mismatches

| Config | `batch_size` | `max_length` | `stride` | `shuffle` |
|---|---|---|---|---|
| Debug | 1 | 4 | 1 | False |
| Training | 8-32 | 256-1024 | 128-512 | True |
| Evaluation | 8 | 256 | 256 | False |

#### Token Embedding Layer

`nn.Embedding(vocab_size, embed_dim)` is a learnable lookup table.
Indexing with token ID `i` returns row `i` of a weight matrix of shape
`(vocab_size, embed_dim)`. These weights are randomly initialised and updated
during LLM training. The result for a batch of shape `(B, S)` is a tensor
of shape `(B, S, D)`.

---

### Stage 4 - Positional Embeddings

Transformer self-attention is **permutation-invariant** — it treats its inputs
as a set rather than an ordered sequence. Without explicit position information,
the model cannot distinguish between `[A, B, C]` and `[C, A, B]`.

**Absolute learned positional embeddings** (the GPT-2 approach) use a second
`nn.Embedding` layer of shape `(context_length, embed_dim)`. Each position index
`0, 1, ..., context_length - 1` maps to its own learned vector, which is
updated alongside the token embeddings during training.

Input to this layer is always the fixed vector `torch.arange(max_length)`.
The output shape is `(S, D)`.

---

### Stage 5 - Final Input Embeddings

The two embedding tensors are combined by element-wise addition:

```python
input_embeddings = token_embeddings + pos_embeddings
# (B, S, D) + (S, D)  ->  (B, S, D)   via broadcasting
```

PyTorch broadcasts `pos_embeddings` of shape `(S, D)` across all `B` samples in
the batch automatically. The resulting tensor of shape `(B, S, D)` encodes both
*what* each token is and *where* it appears in the sequence — exactly what the
first Transformer layer expects as input.

---

## Hyperparameter Reference

| Name | Notebook value | Typical GPT-2 | Effect |
|---|---|---|---|
| `VOCAB_SIZE` | 50,257 | 50,257 | Rows in the token embedding table |
| `EMBED_DIM` | 256 | 768 | Width of every embedding vector |
| `MAX_LENGTH` | 4 | 1,024 | Tokens the model sees per forward pass |
| `stride` | 4 | 512 | Overlap between consecutive training windows |
| `batch_size` | 8 | 512 | Sequences processed per gradient update |

---

## Dependencies

| Package | Purpose |
|---|---|
| `tiktoken` | OpenAI's BPE tokenizer |
| `torch` | Tensor ops, `Dataset`, `DataLoader`, `nn.Embedding` |
| `re` (stdlib) | Regex-based tokenisation for V1/V2 |

---

## References

- [tiktoken — OpenAI](https://github.com/openai/tiktoken)
- [GPT-2 Paper — Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Byte Pair Encoding — Sennrich et al., 2016](https://arxiv.org/abs/1508.07909)
- [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
- [PyTorch DataLoader Docs](https://pytorch.org/docs/stable/data.html)

---

## License

MIT — free to use, adapt, and share.
