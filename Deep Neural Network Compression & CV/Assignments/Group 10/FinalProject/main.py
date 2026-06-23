import os
import torch
import numpy as np

from models import mnist_model
from data import MNIST_loader
from utils.training import evaluate, train_and_eval
from utils.test_eval import (
    measure_model_size,
    measure_runtime_memory,
    measure_system_ram
)
from compression.prune import prune_model
from compression.quantization import quantize_model
from compression.huffman import save_huffman_encoded
from utils.loading import save_model_npz, load_model_from_npz
from config import config_device


# ---------------- CONFIG ----------------
path = "data"
device = config_device()

# Ensure output folder exists
os.makedirs("compressed_models", exist_ok=True)


# ---------------- LOAD DATA ----------------
train_loader, test_loader = MNIST_loader(path)


# ---------------- CREATE MODEL ----------------
model = mnist_model().to(device)


# ---------------- BASELINE TRAINING ----------------
print("\n=== BASELINE TRAINING ===")
train_and_eval(model, train_loader, test_loader, device, epochs=5)


# ---------------- PRUNING ----------------
print("\n=== PRUNING ===")
prune_model(model, 0.98)

# Fine-tune after pruning
train_and_eval(model, train_loader, test_loader, device, epochs=5)


# ---------------- SPARSITY CHECK ----------------
def count_mask_sparsity(model):
    total = 0
    zeros = 0

    for module in model.modules():
        if hasattr(module, "mask") and module.mask is not None:
            total += module.mask.numel()
            zeros += (module.mask == 0).sum().item()

    if total == 0:
        print("Sparsity Check: No pruned layers found.")
    else:
        print(f"Masked sparsity: {100 * zeros / total:.2f}%")


count_mask_sparsity(model)


# Save pruned model
torch.save(model.state_dict(), "compressed_models/model.pth")


# ---------------- QUANTIZATION ----------------
print("\n=== QUANTIZATION ===")
quantize_model(model, 4)
print("Model quantized to 4 clusters!")

# Fine-tune after quantization
train_and_eval(model, train_loader, test_loader, device, epochs=5)


# ---------------- HUFFMAN ENCODING ----------------
print("\n=== HUFFMAN ENCODING ===")

all_cluster_maps = []

for module in model.modules():
    if hasattr(module, "cluster_map") and module.cluster_map is not None:
        cluster_map_flat = module.cluster_map.detach().cpu().numpy().flatten()
        all_cluster_maps.extend(cluster_map_flat)

all_cluster_maps = np.array(all_cluster_maps, dtype=np.uint8)

print(f"Collected cluster maps: {len(all_cluster_maps)}")

if len(all_cluster_maps) > 0:
    save_huffman_encoded(
        all_cluster_maps,
        "compressed_models/huffman_encoded.bin"
    )
else:
    print("Warning: No cluster maps found. Skipping Huffman encoding.")


# ---------------- SAVE NPZ MODEL ----------------
print("\n=== SAVING COMPRESSED NPZ MODEL ===")
npz_path = "compressed_models/compressed.npz"
save_model_npz(model, npz_path)


# ---------------- LOAD COMPRESSED MODEL ----------------
print("\n=== LOADING COMPRESSED MODEL ===")
model2 = mnist_model()
model2 = load_model_from_npz(model2, npz_path, device)
model2 = model2.to(device)

compressed_acc = evaluate(model2, test_loader, device)
print(f"Compressed Model Accuracy: {compressed_acc:.2f}%")


# ---------------- FINAL METRICS ----------------
print("\n--- FINAL PROJECT METRICS ---")

size_normal = measure_model_size("compressed_models/model.pth")
size_compressed = measure_model_size("compressed_models/compressed.npz")

# Huffman file may not exist if skipped
huffman_path = "compressed_models/huffman_encoded.bin"
size_huffman = 0
if os.path.exists(huffman_path):
    size_huffman = measure_model_size(huffman_path)
else:
    print("No Huffman file generated.")


# NPZ compression ratio
if size_compressed > 0:
    ratio_npz = size_normal / size_compressed
    print(f"\nNPZ Compression Ratio: {ratio_npz:.2f}x")


# Huffman compression ratio
if size_huffman > 0:
    ratio_huffman = size_normal / size_huffman
    print(f"Huffman Compression Ratio: {ratio_huffman:.2f}x")


# ---------------- MEMORY REPORT ----------------
measure_runtime_memory(device)
measure_system_ram()