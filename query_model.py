import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


import transformers.models.gpt2.modeling_gpt2 as gpt2_modeling
import transformers.models.auto.auto_factory as auto_factory
from transformers.models.auto.auto_factory import _get_model_class
from ttl_gpt2 import (
    GPT2BlockTTL,
    GPT2AttentionTTL,
    GPT2MLPTTL,
    GPT2ModelTTL,
    GPT2LMHeadModelTTL,
)


from ttl_activations import NewGELUActivationTTL
from ttl_pytorch_utils import Conv1DTTL
from ttl_sdpa_attention import sdpa_attention_forward_ttl


def _get_model_class_override(config, model_mapping):
    print("Working with TTL model")
    return GPT2LMHeadModelTTL


tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")

text = "Once upon a time"
inputs = tokenizer(text, return_tensors="pt")

# Method 1: Load config, modify it, then pass to from_pretrained
config = AutoConfig.from_pretrained("sshleifer/tiny-gpt2")
config.run_ttl = False
config.use_cache = False


# Load model with overridden config
model = AutoModelForCausalLM.from_pretrained(
    "sshleifer/tiny-gpt2",
    config=config,  # Pass the modified config
    ignore_mismatched_sizes=True,  # Useful if you change vocab_size or other size-related params
)
model.eval()

# Pre-compute inputs_embeds from input_ids using model's embedding layer
with torch.no_grad():
    inputs_embeds = model.transformer.wte(inputs["input_ids"])
inputs_for_embeds = {
    "inputs_embeds": inputs_embeds,
    "attention_mask": inputs["attention_mask"],
}

# # Export model to ONNX
# onnx_output_path = "model.onnx"


# # Wrapper to avoid past_key_values arg mismatch (model expects Cache, tracer may pass Tensor)
# class OnnxExportWrapper(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, inputs_embeds, attention_mask):
#         return self.model(
#             inputs_embeds=inputs_embeds,
#             attention_mask=attention_mask,
#             past_key_values=None,
#             use_cache=False,
#         )


# wrapped_model = OnnxExportWrapper(model)
# torch.onnx.export(
#     wrapped_model,
#     args=(inputs_embeds, inputs["attention_mask"]),
#     f=onnx_output_path,
#     input_names=["inputs_embeds", "attention_mask"],
#     output_names=["logits"],
#     opset_version=17,
#     do_constant_folding=True,
# )
# print(f"Model exported to {onnx_output_path}")

with torch.no_grad():
    outputs = model(**inputs_for_embeds)


# overriding functionalities
gpt2_modeling.GPT2LMHeadModel = GPT2LMHeadModelTTL
gpt2_modeling.GPT2Model = GPT2ModelTTL
gpt2_modeling.Conv1D = Conv1DTTL
gpt2_modeling.GPT2MLP = GPT2MLPTTL
gpt2_modeling.GPT2Block = GPT2BlockTTL
gpt2_modeling.GPT2Attention = GPT2AttentionTTL
gpt2_modeling.ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_attention_forward_ttl
gpt2_modeling.ACT2FN["gelu_new"] = NewGELUActivationTTL
auto_factory._get_model_class = _get_model_class_override


config.run_ttl = True  # Use attribute assignment, not config["run_ttl"]
model_ttl = AutoModelForCausalLM.from_pretrained(
    "sshleifer/tiny-gpt2",
    config=config,  # Pass the modified config
    ignore_mismatched_sizes=True,  # Useful if you change vocab_size or other size-related params
)
model_ttl.eval()


with torch.no_grad():
    outputs_ttl = model_ttl(**inputs_for_embeds)

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
        are_close = np.allclose(logits_orig_np, logits_ttl_np, atol=1e-2)
        are_equal = np.array_equal(logits_orig_np, logits_ttl_np)

        print(f"\nComparison:")
        print(f"  Arrays are equal (exact):     {are_equal}")
        print(f"  Arrays are close (atol=1e-2): {are_close}")

        # Find positions with largest differences
        if max_diff > 1e-6:
            max_diff_indices = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"\nLargest difference at position: {max_diff_indices}")
            print(f"  Original value: {logits_orig_np[max_diff_indices]:.6e}")
            print(f"  TTL value:      {logits_ttl_np[max_diff_indices]:.6e}")
            print(f"  Difference:     {diff[max_diff_indices]:.6e}")

        # Compare top predictions
        print(f"\nTop-10 predictions comparison:")
        orig_top10 = torch.topk(logits_original[0, -1], 10)
        ttl_top10 = torch.topk(logits_ttl[0, -1], 10)

        print(f"  Original top-10 indices: {orig_top10.indices.tolist()}")
        print(f"  TTL top-10 indices:      {ttl_top10.indices.tolist()}")
        print(
            f"  Top-10 indices match:    {torch.equal(orig_top10.indices, ttl_top10.indices)}"
        )
        print(
            f"  Original top-10 values: {[f'{v:.3f}' for v in orig_top10.values.tolist()]}"
        )
        print(
            f"  TTL top-10 values:      {[f'{v:.3f}' for v in ttl_top10.values.tolist()]}"
        )

    else:
        print("  ✗ Shapes do NOT match!")

logits = outputs.logits
next_token = torch.argmax(logits[0, -1])
print(tokenizer.decode(next_token))
