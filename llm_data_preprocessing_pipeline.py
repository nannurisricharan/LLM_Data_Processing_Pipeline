# =============================================================================
#  LLM Data Preprocessing Pipeline
#  Raw Text --> Tokenized Text --> Token IDs -->
#  Token Embeddings --> Positional Embeddings --> Input Embeddings
# =============================================================================
#
#  Pipeline Overview
#  -----------------
#  Stage 1 : Raw Text           --> Tokenized Text    (re.split)
#  Stage 2 : Tokenized Text     --> Token IDs         (vocab dict / BPE tiktoken)
#  Stage 3 : Token IDs          --> Token Embeddings  (nn.Embedding + DataLoader)
#  Stage 4 : Token Embeddings   --> Positional Embeddings (nn.Embedding)
#  Stage 5 : Token + Positional --> Input Embeddings  (element-wise addition)
#
#  Requirements : pip install tiktoken torch
#  Usage        : python llm_data_preprocessing_pipeline.py
# =============================================================================


# -- Imports -------------------------------------------------------------------

import re
import importlib

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


# -- Version check -------------------------------------------------------------

print("=" * 60)
print("  LLM Data Preprocessing Pipeline")
print("=" * 60)
print(f"  PyTorch  : {torch.__version__}")
print(f"  tiktoken : {importlib.metadata.version('tiktoken')}")
print()


# -- Load corpus ---------------------------------------------------------------

CORPUS_FILE = "the-verdict.txt"

with open(CORPUS_FILE, "r", encoding="utf-8") as fh:
    raw_text = fh.read()

print(f"[Corpus]  {len(raw_text):,} characters loaded from '{CORPUS_FILE}'")
print(f"          Preview: {raw_text[:100]}")
print()


# =============================================================================
#  STAGE 1 -- Raw Text  -->  Tokenized Text
# =============================================================================
#  Split the character stream into string tokens (one per word / punctuation).
#  Whitespace tokens are discarded -- they carry no semantic value here.
# =============================================================================

print("-" * 60)
print("STAGE 1 -- Raw Text  -->  Tokenized Text")
print("-" * 60)

SPLIT_PATTERN = r"([,.:;?_!()\']|--|[\s])"

tokenized = re.split(SPLIT_PATTERN, raw_text)
tokenized = [tok.strip() for tok in tokenized if tok.strip()]

print(f"  Total tokens : {len(tokenized):,}")
print(f"  First 20     : {tokenized[:20]}")
print()


# =============================================================================
#  STAGE 2 -- Tokenized Text  -->  Token IDs
# =============================================================================
#  Assign a unique integer to every distinct string token.
#
#  Step 2a  Build vocabulary
#           Sort unique tokens, then append two special tokens:
#             <|unk|>         -- fallback for OOV words
#             <|endoftext|>   -- document boundary marker
#
#  Step 2b  TokenizerV1  raises KeyError on OOV (closed-corpus use)
#           TokenizerV2  maps OOV to <|unk|>   (open-domain / production)
#
#  Step 2c  BPE via tiktoken -- sub-word encoding, no <|unk|> needed
# =============================================================================

print("-" * 60)
print("STAGE 2 -- Tokenized Text  -->  Token IDs")
print("-" * 60)

# -- 2a. Vocabulary --
unique_tokens = sorted(set(tokenized))
unique_tokens += ["<|endoftext|>", "<|unk|>"]
vocab      = {token: idx for idx, token in enumerate(unique_tokens)}
vocab_size = len(vocab)

print(f"  Vocabulary size : {vocab_size:,} tokens")
print(f"  Last 6 entries  : {list(vocab.items())[-6:]}")


# -- 2b-i. TokenizerV1 --
class TokenizerV1:
    """
    Bidirectional vocab tokenizer.  Raises KeyError on OOV words.
    Best for closed-corpus scenarios where all tokens are guaranteed known.
    """

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {idx: tok for tok, idx in vocab.items()}

    def encode(self, text):
        """Convert a text string to a list of integer token IDs."""
        tokens = re.split(SPLIT_PATTERN, text)
        tokens = [t.strip() for t in tokens if t.strip()]
        return [self.str_to_int[t] for t in tokens]   # KeyError on OOV

    def decode(self, ids):
        """Convert integer token IDs back to a text string."""
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r"\s+([,.?!()'])", r"\1", text)


tok_v1  = TokenizerV1(vocab)
passage = ('"It\'s the last he painted, you know,"'
           ' Mrs. Gisburn said with pardonable pride.')
ids_v1  = tok_v1.encode(passage)
print(f"  [V1] Encoded : {ids_v1}")
print(f"  [V1] Decoded : {tok_v1.decode(ids_v1)}")


# -- 2b-ii. TokenizerV2 --
class TokenizerV2:
    """
    OOV-tolerant tokenizer.
    Replaces any unseen word with the <|unk|> special token.
    Also supports <|endoftext|> document boundary marking.
    """

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {idx: tok for tok, idx in vocab.items()}

    def encode(self, text):
        """Encode text, mapping unknown words to <|unk|>."""
        tokens = re.split(SPLIT_PATTERN, text)
        tokens = [t.strip() for t in tokens if t.strip()]
        tokens = [t if t in self.str_to_int else "<|unk|>" for t in tokens]
        return [self.str_to_int[t] for t in tokens]

    def decode(self, ids):
        """Decode token IDs back to text."""
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r"\s+([,.:;?!()'])", r"\1", text)


tok_v2 = TokenizerV2(vocab)
doc_a  = "Hello, do you like tea?"
doc_b  = "In the sunlit terraces of the palace."
joined = " <|endoftext|> ".join([doc_a, doc_b])

print(f"  [V2] Input   : {joined}")
print(f"  [V2] IDs     : {tok_v2.encode(joined)}")
print(f"  [V2] Decoded : {tok_v2.decode(tok_v2.encode(joined))}")


# -- 2c. BPE via tiktoken --
bpe_tokenizer = tiktoken.get_encoding("gpt2")
print(f"\n  [BPE] Vocabulary size : {bpe_tokenizer.n_vocab:,}")

sample_text = (
    "Hello, do you like tea? <|endoftext|> "
    "In the sunlit terraces of someunknownPlace."
)
bpe_ids = bpe_tokenizer.encode(sample_text, allowed_special={"<|endoftext|>"})
print(f"  [BPE] Token IDs : {bpe_ids}")
print(f"  [BPE] Decoded   : {bpe_tokenizer.decode(bpe_ids)}")

novel_ids = bpe_tokenizer.encode("Akwirw ier")
print(f"\n  [BPE] Akwirw ier -> IDs    : {novel_ids}")
print(f"  [BPE] Akwirw ier -> pieces : {[bpe_tokenizer.decode([i]) for i in novel_ids]}")
print()


# =============================================================================
#  STAGE 3 -- Token IDs  -->  Token Embeddings  (+  DataLoader)
# =============================================================================
#  Sliding-Window Input-Target Pairs
#  -----------------------------------
#  Next-token prediction training requires (input, target) tensor pairs.
#  A window of width max_length steps through the token sequence:
#
#    tokens  : [t0, t1, t2, t3, t4, ...]
#    input x : [t0, t1, t2, t3]
#    target y: [t1, t2, t3, t4]    <- shifted right by 1
#
#  stride = 1          -> max overlap, max data reuse
#  stride = max_length -> no overlap, cleanest batches
#
#  GPTDatasetV1         torch.utils.data.Dataset with sliding-window logic
#  create_dataloader_v1 DataLoader factory (batching, shuffling, drop_last)
#  nn.Embedding         learnable lookup table: token ID -> dense vector
# =============================================================================

print("-" * 60)
print("STAGE 3 -- Token IDs  -->  Token Embeddings  (+  DataLoader)")
print("-" * 60)


class GPTDatasetV1(Dataset):
    """
    Sliding-window dataset for next-token prediction training.

    Parameters
    ----------
    txt        : full corpus as a single string
    tokenizer  : tiktoken-compatible encoder
    max_length : context window width (sequence length)
    stride     : tokens to advance between consecutive windows
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        all_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        self.input_ids  = []
        self.target_ids = []

        for start in range(0, len(all_ids) - max_length, stride):
            self.input_ids.append(
                torch.tensor(all_ids[start : start + max_length])
            )
            self.target_ids.append(
                torch.tensor(all_ids[start + 1 : start + max_length + 1])
            )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size  = 4,
    max_length  = 256,
    stride      = 128,
    shuffle     = True,
    drop_last   = True,
    num_workers = 0,
):
    """
    Wrap GPTDatasetV1 in a PyTorch DataLoader.

    batch_size   : sequences per mini-batch
    max_length   : context window size
    stride       : step between windows (< max_length => overlapping)
    shuffle      : randomise order each epoch -- recommended for training
    drop_last    : discard final incomplete batch (prevents shape mismatches)
    num_workers  : parallel worker processes for data loading
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset   = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        drop_last   = drop_last,
        num_workers = num_workers,
    )


# -- Verify DataLoader --
dl_debug = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)
it      = iter(dl_debug)
batch_a = next(it)
batch_b = next(it)

print(f"  [DataLoader] Batch A  inputs: {batch_a[0]}  targets: {batch_a[1]}")
print(f"  [DataLoader] Batch B  inputs: {batch_b[0]}  targets: {batch_b[1]}")
print("               (stride=1: Batch B starts at Batch A's 2nd token)")


# -- Token Embedding Layer --
VOCAB_SIZE = 50_257   # GPT-2 BPE vocabulary size
EMBED_DIM  = 256      # embedding dimension (GPT-3 uses 12,288; 256 for demo)
MAX_LENGTH = 4        # context window / sequence length

torch.manual_seed(42)
token_embedding_layer = torch.nn.Embedding(VOCAB_SIZE, EMBED_DIM)

# Non-overlapping batch (stride = max_length = 4)
dl_train = create_dataloader_v1(
    raw_text, batch_size=8, max_length=MAX_LENGTH,
    stride=MAX_LENGTH, shuffle=False,
)
inputs, targets = next(iter(dl_train))

print(f"\n  Token ID tensor shape  : {inputs.shape}")   # (8, 4)
token_embeddings = token_embedding_layer(inputs)
print(f"  Token embeddings shape : {token_embeddings.shape}")  # (8, 4, 256)
print()


# =============================================================================
#  STAGE 4 -- Token Embeddings  -->  Positional Embeddings
# =============================================================================
#  Transformer self-attention is permutation-invariant: it cannot distinguish
#  [A,B,C] from [C,A,B] without explicit position information.
#
#  Absolute learned positional embeddings (GPT-2 style):
#  A second nn.Embedding of shape (context_length, embed_dim) maps each
#  position index 0..context_length-1 to a learned vector updated at training.
#  Input is always the fixed sequence torch.arange(max_length).
# =============================================================================

print("-" * 60)
print("STAGE 4 -- Token Embeddings  -->  Positional Embeddings")
print("-" * 60)

pos_embedding_layer = torch.nn.Embedding(MAX_LENGTH, EMBED_DIM)

# Always pass [0, 1, 2, ..., MAX_LENGTH-1] as position indices
position_indices = torch.arange(MAX_LENGTH)                 # shape: (4,)
pos_embeddings   = pos_embedding_layer(position_indices)    # shape: (4, 256)

print(f"  Position index shape       : {position_indices.shape}")
print(f"  Positional embeddings shape: {pos_embeddings.shape}")
print("  Each row encodes one position within the context window.")
print()


# =============================================================================
#  STAGE 5 -- Final Input Embeddings
# =============================================================================
#  Combine token identity and positional context via element-wise addition.
#  PyTorch broadcasts pos_embeddings (S, D) across all B batch samples:
#
#    (B, S, D) + (S, D)  ->  (B, S, D)   via broadcasting
#
#  Each resulting vector encodes both *what* the token is and *where* it
#  appears. This tensor is the direct input to the first Transformer layer.
# =============================================================================

print("-" * 60)
print("STAGE 5 -- Final Input Embeddings")
print("-" * 60)

# (8, 4, 256) + (4, 256)  ->  (8, 4, 256) via broadcasting
input_embeddings = token_embeddings + pos_embeddings

print(f"  Token emb shape : {token_embeddings.shape}")    # (8, 4, 256)
print(f"  Pos emb shape   : {pos_embeddings.shape}")      # (4, 256) <- broadcast
print(f"  Input emb shape : {input_embeddings.shape}")    # (8, 4, 256)
print()
print("=" * 60)
print("  input_embeddings is ready for the Transformer.")
print("=" * 60)


# =============================================================================
#  PIPELINE SUMMARY
#
#  Stage  Input                     Output                Key tool
#  -----  ------------------------  --------------------  --------------------
#  1      Raw text string           list[str] tokens      re.split
#  2      list[str] tokens          list[int] IDs         vocab / tiktoken BPE
#  3      Batched IDs (B, S)        Float tensor (B,S,D)  nn.Embedding + DL
#  4      Position indices (S,)     Float tensor (S,D)    nn.Embedding
#  5      Token emb + Pos emb       Float tensor (B,S,D)  Element-wise +
#
#  KEY HYPERPARAMETERS
#  Name        Demo    GPT-2   Effect
#  ----------  ------  ------  -----------------------------------------
#  VOCAB_SIZE  50,257  50,257  Rows in the token embedding table
#  EMBED_DIM      256     768  Width of every embedding vector
#  MAX_LENGTH       4   1,024  Tokens per forward pass (context window)
#  stride           4     512  Overlap between consecutive training windows
#  batch_size       8     512  Sequences processed per gradient update
# =============================================================================
