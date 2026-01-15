#!/usr/bin/env python3
"""
Evaluation script for trained GPT-2 model.

This script evaluates a trained GPT-2 model on various metrics including:
- Perplexity
- Text generation quality
- Token-level accuracy
"""

import argparse
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import json
from typing import List, Dict, Optional

# Add transformers to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "transformers"))


class TextDataset(Dataset):
    """Simple dataset for text evaluation."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }


def calculate_perplexity(
    model, tokenizer, texts: List[str], device: str = "cuda", batch_size: int = 8
):
    """
    Calculate perplexity on a list of texts.

    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        texts: List of texts to evaluate
        device: Device to run on
        batch_size: Batch size for evaluation

    Returns:
        Average perplexity
    """
    model.eval()
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating perplexity"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Shift input_ids for labels
            labels = input_ids.clone()

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            # Count non-padding tokens
            num_tokens = attention_mask.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = np.exp(avg_loss)

    return perplexity, avg_loss


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = "cuda",
    num_samples: int = 1,
):
    """
    Generate text from a prompt.

    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        prompt: Input prompt
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling
        device: Device to run on
        num_samples: Number of samples to generate

    Returns:
        List of generated texts
    """
    model.eval()
    generated_texts = []

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_texts.append(generated_text)

    return generated_texts


def evaluate_token_accuracy(
    model, tokenizer, texts: List[str], device: str = "cuda", batch_size: int = 8
):
    """
    Calculate token-level accuracy (next token prediction).

    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        texts: List of texts to evaluate
        device: Device to run on
        batch_size: Batch size for evaluation

    Returns:
        Token accuracy
    """
    model.eval()
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating token accuracy"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get predictions for next tokens
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift to align predictions with next tokens
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()

            # Get predicted tokens
            predicted_tokens = shift_logits.argmax(dim=-1)

            # Calculate accuracy (only on non-padding tokens)
            correct = (predicted_tokens == shift_labels) & shift_mask.bool()
            total_correct += correct.sum().item()
            total_tokens += shift_mask.sum().item()

    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    return accuracy


def load_model_and_tokenizer(
    model_path: str, tokenizer_path: Optional[str] = None, device: str = "cuda"
):
    """
    Load model and tokenizer from paths.

    Args:
        model_path: Path to model checkpoint or directory
        tokenizer_path: Path to tokenizer (if None, uses model_path)
        device: Device to load model on

    Returns:
        model, tokenizer
    """
    # Resolve paths to absolute
    model_path = os.path.abspath(os.path.expanduser(model_path))
    tokenizer_path = tokenizer_path or model_path
    tokenizer_path = os.path.abspath(os.path.expanduser(tokenizer_path))

    print(f"Loading tokenizer from {tokenizer_path}...")
    try:
        # Only try to load from path if it's a directory
        if os.path.isdir(tokenizer_path):
            tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        else:
            # If it's a file, use default tokenizer
            raise FileNotFoundError("Tokenizer path is a file, using default")
    except:
        # Fallback to default GPT-2 tokenizer
        print("Using default GPT-2 tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {model_path}...")

    # Check if it's a checkpoint file or directory
    if os.path.isfile(model_path):
        # Load from checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Try to infer config from checkpoint or use default
        if "config" in checkpoint:
            config = checkpoint["config"]
            if isinstance(config, dict):
                config = GPT2Config(**config)
            elif not isinstance(config, GPT2Config):
                config = GPT2Config()
        else:
            config = GPT2Config()

        model = GPT2LMHeadModel(config)
        model.load_state_dict(state_dict, strict=False)
    elif os.path.isdir(model_path):
        # Load from directory (HuggingFace format)
        model = GPT2LMHeadModel.from_pretrained(model_path)
    else:
        raise FileNotFoundError(
            f"Model path not found: {model_path}. "
            "Please provide a valid checkpoint file (.pt, .pth, .ckpt) or a directory containing model files."
        )

    model.to(device)
    model.eval()

    return model, tokenizer


def load_evaluation_data(data_path: str, max_samples: Optional[int] = None):
    """
    Load evaluation data from file.

    Supports:
    - Text file (one text per line)
    - JSON file with 'texts' key or list of dicts with 'text' key

    Args:
        data_path: Path to data file
        max_samples: Maximum number of samples to load

    Returns:
        List of texts
    """
    print(f"Loading evaluation data from {data_path}...")

    if data_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            texts = [
                item["text"] if isinstance(item, dict) else str(item) for item in data
            ]
        elif isinstance(data, dict) and "texts" in data:
            texts = data["texts"]
        else:
            raise ValueError("JSON file must contain a list or a dict with 'texts' key")
    else:
        # Assume text file
        with open(data_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    if max_samples:
        texts = texts[:max_samples]

    print(f"Loaded {len(texts)} texts for evaluation")
    return texts


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained GPT-2 model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint or directory",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer (defaults to model_path)",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default=None,
        help="Path to evaluation data file (text or JSON)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--test_generation",
        action="store_true",
        help="Test text generation with sample prompts",
    )
    parser.add_argument(
        "--generation_prompts",
        type=str,
        nargs="+",
        default=["The quick brown fox", "Once upon a time", "In the future"],
        help="Prompts for text generation testing",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.tokenizer_path, args.device
    )

    print(f"Model loaded on {args.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    results = {}

    # Evaluate on provided data
    if args.eval_data:
        texts = load_evaluation_data(args.eval_data, args.max_samples)

        print("\n" + "=" * 50)
        print("Evaluating on provided data...")
        print("=" * 50)

        # Calculate perplexity
        print("\nCalculating perplexity...")
        perplexity, avg_loss = calculate_perplexity(
            model, tokenizer, texts, args.device, args.batch_size
        )
        results["perplexity"] = perplexity
        results["average_loss"] = avg_loss
        print(f"Perplexity: {perplexity:.4f}")
        print(f"Average Loss: {avg_loss:.4f}")

        # Calculate token accuracy
        print("\nCalculating token accuracy...")
        accuracy = evaluate_token_accuracy(
            model, tokenizer, texts, args.device, args.batch_size
        )
        results["token_accuracy"] = accuracy
        print(f"Token Accuracy: {accuracy:.4%}")

    # Test text generation
    if args.test_generation:
        print("\n" + "=" * 50)
        print("Testing text generation...")
        print("=" * 50)

        generation_results = {}
        for prompt in args.generation_prompts:
            print(f"\nPrompt: '{prompt}'")
            generated = generate_text(
                model,
                tokenizer,
                prompt,
                max_length=100,
                temperature=0.8,
                device=args.device,
                num_samples=3,
            )
            generation_results[prompt] = generated
            for i, text in enumerate(generated, 1):
                print(f"  Sample {i}: {text[:200]}...")

        results["generation_samples"] = generation_results

    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Evaluation complete! Results saved to {results_path}")
    print(f"{'='*50}")

    # Print summary
    print("\nSummary:")
    for key, value in results.items():
        if key != "generation_samples":
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
