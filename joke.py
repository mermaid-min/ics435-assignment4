"""
Fine-Tuning GPT-2 for Joke Generation
Dataset: 1,622 short jokes (data)
Task: Generate a full joke given the first 3 words as a prompt.

Requirements:
    pip install transformers datasets torch accelerate

Usage:
    # Train the model
    python gpt2_joke_finetune.py --train

    # Generate jokes after training
    python gpt2_joke_finetune.py --generate --prompt "What did the"
    python gpt2_joke_finetune.py --generate --prompt "Why did the"
"""

import os
import csv
import argparse
import random
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    AdamW,
    get_linear_schedule_with_warmup,
)


# Configuration
CONFIG = {
    # Paths
    "data_path": "data",          # dataset
    "output_dir": "gpt2_jokes_model",       # Where the model is saved

    # Model
    "model_name": "gpt2",                   # Base GPT-2 (117M params); use "gpt2-medium" for better results

    # Training hyperparameters
    "max_length": 128,                      # Max token length per joke
    "batch_size": 8,                        # Reduce to 4 if you hit GPU memory issues
    "epochs": 5,                            # More epochs = better quality (try 10–15 for best results)
    "learning_rate": 5e-5,
    "warmup_steps": 100,
    "gradient_accumulation_steps": 2,       # Simulates batch_size * 2 = 16 effective batch
    "val_split": 0.1,                       # 10% of data for validation

    # Generation
    "max_gen_length": 100,
    "temperature": 0.9,                     # Higher = more creative/random
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 3,              # Generate 3 joke candidates

    # Special tokens
    "bos_token": "<|startoftext|>",
    "eos_token": "<|endoftext|>",
    "pad_token": "<|pad|>",
}


# 1. Data Loading & Preprocessing

def load_jokes(csv_path: str) -> list[str]:
    """Load jokes from the CSV file."""
    jokes = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            joke = row["Joke"].strip()
            if joke:
                jokes.append(joke)
    print(f"Loaded {len(jokes)} jokes from '{csv_path}'")
    return jokes


def format_joke(joke: str, bos: str, eos: str) -> str:
    """
    Wrap each joke with special tokens so GPT-2 learns where jokes start/end.
    Format: <|startoftext|> JOKE <|endoftext|>
    """
    return f"{bos} {joke} {eos}"


def train_val_split(jokes: list[str], val_fraction: float = 0.1, seed: int = 42):
    """Split jokes into train/validation sets."""
    random.seed(seed)
    shuffled = jokes.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_fraction))
    return shuffled[:split_idx], shuffled[split_idx:]


# 2. PyTorch Dataset

class JokeDataset(Dataset):
    """
    Tokenizes jokes and creates input_ids + labels for causal language modeling.
    Labels = input_ids (GPT-2 predicts next token at every position).
    Padding tokens are masked to -100 so they don't contribute to loss.
    """

    def __init__(self, jokes: list[str], tokenizer: GPT2Tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for joke in jokes:
            encodings = tokenizer(
                joke,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].squeeze()
            attention_mask = encodings["attention_mask"].squeeze()

            # Labels = input_ids, but padding positions → -100 (ignored by CrossEntropyLoss)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# 3. Model & Tokenizer Setup

def setup_tokenizer(model_name: str) -> GPT2Tokenizer:
    """Load GPT-2 tokenizer and add special tokens."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Since GPT-2 has no pad token by default add one
    special_tokens = {
        "bos_token": CONFIG["bos_token"],
        "eos_token": CONFIG["eos_token"],
        "pad_token": CONFIG["pad_token"],
    }
    tokenizer.add_special_tokens(special_tokens)
    print(f" Tokenizer loaded. Vocab size: {len(tokenizer)}")
    return tokenizer


def setup_model(model_name: str, tokenizer: GPT2Tokenizer) -> GPT2LMHeadModel:
    """Load GPT-2 and resize embeddings for new special tokens."""
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))  # Account for added special tokens
    print(f"Model loaded: {model_name} ({sum(p.numel() for p in model.parameters()):,} parameters)")
    return model


# 4. Training Loop

def train(cfg: dict = CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n Training on: {device}")

    # --- Data ---
    jokes_raw = load_jokes(cfg["data_path"])
    formatted = [format_joke(j, cfg["bos_token"], cfg["eos_token"]) for j in jokes_raw]
    train_jokes, val_jokes = train_val_split(formatted, cfg["val_split"])
    print(f"   Train: {len(train_jokes)} | Val: {len(val_jokes)}")

    # --- Tokenizer & Model ---
    tokenizer = setup_tokenizer(cfg["model_name"])
    model = setup_model(cfg["model_name"], tokenizer)
    model.to(device)

    # --- Datasets & Loaders ---
    train_dataset = JokeDataset(train_jokes, tokenizer, cfg["max_length"])
    val_dataset   = JokeDataset(val_jokes,   tokenizer, cfg["max_length"])

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg["batch_size"], shuffle=False)

    # --- Optimizer & Scheduler ---
    total_steps = (len(train_loader) // cfg["gradient_accumulation_steps"]) * cfg["epochs"]
    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["warmup_steps"],
        num_training_steps=total_steps,
    )

    # --- Training ---
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Starting training for {cfg['epochs']} epochs")
    print(f"  Steps per epoch: {len(train_loader)}")
    print(f"{'='*55}\n")

    for epoch in range(1, cfg["epochs"] + 1):
        # ── Train ──
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / cfg["gradient_accumulation_steps"]
            loss.backward()
            total_train_loss += outputs.loss.item()

            if (step + 1) % cfg["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 50 == 0:
                avg = total_train_loss / (step + 1)
                print(f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} | Loss: {avg:.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(f"\n📊 Epoch {epoch}/{cfg['epochs']}  |  Train Loss: {avg_train_loss:.4f}  |  Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(output_dir / "best")
            tokenizer.save_pretrained(output_dir / "best")
            print(f"   💾 Best model saved (val_loss={best_val_loss:.4f})\n")

    # Save final model + training history
    model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"   Models saved to '{output_dir}/'")
    return model, tokenizer


# 5. Joke Generation

def generate_joke(
    prompt: str,
    model_dir: str = None,
    cfg: dict = CONFIG,
) -> list[str]:
    """
    Generate jokes given a prompt (ideally the first 3 words of a joke).

    Args:
        prompt: e.g. "What did the" or "Why don't scientists"
        model_dir: Path to saved model directory (defaults to best model)
    """
    if model_dir is None:
        model_dir = str(Path(cfg["output_dir"]) / "best")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from '{model_dir}'...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Prepend the BOS token + prompt (mirrors training format)
    full_prompt = f"{cfg['bos_token']} {prompt}"
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=cfg["max_gen_length"],
            temperature=cfg["temperature"],
            top_k=cfg["top_k"],
            top_p=cfg["top_p"],
            do_sample=True,
            num_return_sequences=cfg["num_return_sequences"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,         # Discourages repeating phrases
        )

    jokes = []
    for ids in output_ids:
        text = tokenizer.decode(ids, skip_special_tokens=True)
        # Strip the prompt prefix to isolate the generated part
        if text.lower().startswith(prompt.lower()):
            text = text  # Keep the full joke including prompt
        jokes.append(text.strip())

    return jokes


# 6. Quick Demo (no GPU needed)

def demo_untrained():
    """Show what the raw GPT-2 generates (before fine-tuning) for comparison."""
    print("\n🔬 Demo: Generating from BASE (untrained) GPT-2 for comparison...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    prompt = "What did the bartender say"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(input_ids, max_length=60, do_sample=True, temperature=0.9)

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"  Prompt : {prompt}")
    print(f"  Output : {result}\n")


# 7. CLI Entry Point

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 for joke generation")
    parser.add_argument("--train",    action="store_true", help="Run training")
    parser.add_argument("--generate", action="store_true", help="Generate jokes")
    parser.add_argument("--demo",     action="store_true", help="Demo base GPT-2 (no training needed)")
    parser.add_argument("--prompt",   type=str, default="What did the",
                        help="Prompt for joke generation (default: 'What did the')")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to fine-tuned model (default: gpt2_jokes_model/best)")
    parser.add_argument("--epochs",   type=int, default=CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--data_path", type=str, default=CONFIG["data_path"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override config from CLI flags
    CONFIG["epochs"]     = args.epochs
    CONFIG["batch_size"] = args.batch_size
    CONFIG["data_path"]  = args.data_path

    if args.demo:
        demo_untrained()

    if args.train:
        train(CONFIG)

    if args.generate:
        jokes = generate_joke(args.prompt, model_dir=args.model_dir)
        print(f"\n🎭 Generated jokes for prompt: \"{args.prompt}\"")
        print("─" * 55)
        for i, joke in enumerate(jokes, 1):
            print(f"  [{i}] {joke}")
        print()

    if not any([args.train, args.generate, args.demo]):
        print(__doc__)
        print("\nExample commands:")
        print("  python gpt2_joke_finetune.py --demo")
        print("  python gpt2_joke_finetune.py --train --epochs 10")
        print('  python gpt2_joke_finetune.py --generate --prompt "Why did the chicken"')