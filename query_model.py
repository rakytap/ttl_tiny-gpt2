import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


import transformers.models.gpt2.modeling_gpt2 as gpt2_modeling
from ttl_gpt2 import GPT2BlockTTL
from ttl_pytorch_utils import Conv1DTTL


tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")

text = "Once upon a time"
inputs = tokenizer(text, return_tensors="pt")

# Method 1: Load config, modify it, then pass to from_pretrained
config = AutoConfig.from_pretrained("sshleifer/tiny-gpt2")
config.run_ttl = False

# Load model with overridden config
model = AutoModelForCausalLM.from_pretrained(
    "sshleifer/tiny-gpt2",
    config=config,  # Pass the modified config
    ignore_mismatched_sizes=True,  # Useful if you change vocab_size or other size-related params
)
model.eval()

with torch.no_grad():
    outputs = model(**inputs)


# overriding functionalities
gpt2_modeling.Conv1D = Conv1DTTL
gpt2_modeling.GPT2Block = GPT2BlockTTL


config.run_ttl = True  # Use attribute assignment, not config["run_ttl"]
model_ttl = AutoModelForCausalLM.from_pretrained(
    "sshleifer/tiny-gpt2",
    config=config,  # Pass the modified config
    ignore_mismatched_sizes=True,  # Useful if you change vocab_size or other size-related params
)
model_ttl.eval()


with torch.no_grad():
    outputs_ttl = model_ttl(**inputs)

# Compare outputs
print("=" * 80)
print("COMPARING OUTPUTS")
print("=" * 80)

# Compare logits (main output)
if hasattr(outputs, "logits") and hasattr(outputs_ttl, "logits"):
    logits_original = outputs.logits
    logits_ttl = outputs_ttl.logits

    print(f"\nLogits shapes:")
    print(f"  Original: {logits_original.shape}")
    print(f"  TTL:      {logits_ttl.shape}")

    # Check if shapes match
    if logits_original.shape == logits_ttl.shape:
        print("  ✓ Shapes match")

        # Convert to numpy for easier comparison
        logits_orig_np = logits_original.detach().cpu().numpy()
        logits_ttl_np = logits_ttl.detach().cpu().numpy()

        # Calculate differences
        diff = np.abs(logits_orig_np - logits_ttl_np)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)

        print(f"\nDifferences:")
        print(f"  Max absolute difference:  {max_diff:.6e}")
        print(f"  Mean absolute difference: {mean_diff:.6e}")
        print(f"  Std of differences:       {std_diff:.6e}")

        # Check if they're close (within numerical precision)
        are_close = np.allclose(logits_orig_np, logits_ttl_np, rtol=1e-5, atol=1e-6)
        are_equal = np.array_equal(logits_orig_np, logits_ttl_np)

        print(f"\nComparison:")
        print(f"  Arrays are equal (exact):     {are_equal}")
        print(f"  Arrays are close (rtol=1e-5): {are_close}")

        # Find positions with largest differences
        if max_diff > 1e-6:
            max_diff_indices = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"\nLargest difference at position: {max_diff_indices}")
            print(f"  Original value: {logits_orig_np[max_diff_indices]:.6e}")
            print(f"  TTL value:      {logits_ttl_np[max_diff_indices]:.6e}")
            print(f"  Difference:     {diff[max_diff_indices]:.6e}")

        # Compare top predictions
        print(f"\nTop-5 predictions comparison:")
        orig_top5 = torch.topk(logits_original[0, -1], 5)
        ttl_top5 = torch.topk(logits_ttl[0, -1], 5)

        print(f"  Original top-5 indices: {orig_top5.indices.tolist()}")
        print(f"  TTL top-5 indices:      {ttl_top5.indices.tolist()}")
        print(
            f"  Top-5 indices match:    {torch.equal(orig_top5.indices, ttl_top5.indices)}"
        )

    else:
        print("  ✗ Shapes do NOT match!")

logits = outputs.logits
next_token = torch.argmax(logits[0, -1])
print(tokenizer.decode(next_token))
