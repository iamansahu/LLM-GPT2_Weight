# GPT-based LLM with GPT-2 Weights

This project implements a **from-scratch GPT-style Transformer model**
in PyTorch and demonstrates:

-   Training on a text corpus
-   Text generation with temperature and top-k sampling
-   Saving and loading model checkpoints
-   Importing pretrained **OpenAI GPT-2 weights** (TensorFlow format)
    into the custom PyTorch model

------------------------------------------------------------------------

## 1. Model Components

### Configuration

``` python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

### Core Modules

-   **LayerNorm**: Custom normalization with learnable scale and shift.
-   **GELU**: Activation function used in GPT-2.
-   **FeedForward**: Two linear layers with GELU in between.
-   **MultiHeadAttention**: Implements scaled dot-product attention with
    causal masking.
-   **TransformerBlock**: One attention + feed-forward block with
    residual connections.
-   **GPTModel**: Stacks multiple TransformerBlocks with token and
    positional embeddings.

------------------------------------------------------------------------

## 2. Tokenization

Uses `tiktoken`: - `text_to_token_ids(text, tokenizer)` → encodes text
into token IDs. - `token_ids_to_text(token_ids, tokenizer)` → decodes
IDs back to text.

------------------------------------------------------------------------

## 3. Dataset and Dataloader

### `GPTDatasetV1`

-   Converts text into overlapping input-target chunks.
-   `input_ids`: sequence of tokens.
-   `target_ids`: shifted by one token (next-token prediction).

### `create_dataloader_v1`

Creates PyTorch `DataLoader` for batching training/validation samples.

------------------------------------------------------------------------

## 4. Training

### Loss Calculation

``` python
calc_loss_batch(input_batch, target_batch, model, device)
```

Uses **cross-entropy** between logits and target tokens.

### Training Loop

``` python
train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer)
```

-   Trains with **AdamW optimizer**
-   Periodically evaluates train/val loss
-   Generates sample text after each epoch

### Evaluation

``` python
evaluate_model(model, train_loader, val_loader, device, eval_iter)
```

------------------------------------------------------------------------

## 5. Text Generation

``` python
generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None)
```

-   **Temperature** controls randomness
-   **Top-k sampling** restricts to top-k probable tokens
-   **Greedy decoding** when temperature = 0

------------------------------------------------------------------------

## 6. Saving and Loading

### Save Model

``` python
torch.save(model.state_dict(), "model.pth")
```

### Save with Optimizer

``` python
torch.save({
  "model_state_dict": model.state_dict(),
  "optimizer_state_dict": optimizer.state_dict(),
}, "model_and_optimizer.pth")
```

### Load

``` python
checkpoint = torch.load("model_and_optimizer.pth")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

------------------------------------------------------------------------

## 7. GPT-2 Weight Import

### Download

``` python
download_and_load_gpt2(model_size="124M", models_dir="gpt2")
```

### Weight Conversion

``` python
load_weights_into_gpt(gpt, params)
```

------------------------------------------------------------------------

## 8. Example Usage

### Train from scratch

``` python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
train_model_simple(model, train_loader, val_loader, optimizer, device,
                   num_epochs=10, eval_freq=5, eval_iter=5,
                   start_context="Every effort moves you", tokenizer=tokenizer)
```

### Generate text

``` python
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=50,
    temperature=1.5
)
print(token_ids_to_text(token_ids, tokenizer))
```

### Load GPT-2 pretrained weights

``` python
settings, params = download_and_load_gpt2("124M", "gpt2")
gpt = GPTModel(NEW_CONFIG)
load_weights_into_gpt(gpt, params)
gpt.eval()
```

------------------------------------------------------------------------

## 9. Results

-   Training from scratch shows decreasing loss and progressively
    coherent text.
-   Imported GPT-2 weights produce fluent completions immediately.
