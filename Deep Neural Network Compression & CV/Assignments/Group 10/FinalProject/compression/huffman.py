# compression/huffman.py

import heapq
from collections import Counter
import pickle


# ---------------- HUFFMAN NODE ----------------
class HuffmanNode:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


# ---------------- BUILD TREE ----------------
def build_huffman_tree(data):
    """
    Build Huffman tree from input data
    Handles:
    - empty input
    - single-symbol input
    """
    if len(data) == 0:
        raise ValueError("Cannot build Huffman tree: input data is empty.")

    freq = Counter(data)

    heap = [HuffmanNode(sym, f) for sym, f in freq.items()]
    heapq.heapify(heap)

    # Special case: only one unique symbol
    if len(heap) == 1:
        root = HuffmanNode(freq=heap[0].freq)
        root.left = heap[0]
        return root

    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)

        merged = HuffmanNode(freq=n1.freq + n2.freq)
        merged.left = n1
        merged.right = n2

        heapq.heappush(heap, merged)

    return heap[0]


# ---------------- GENERATE CODES ----------------
def generate_codes(node, prefix="", codebook=None):
    """
    Generate Huffman binary codes recursively
    """
    if codebook is None:
        codebook = {}

    if node is None:
        return codebook

    if node.symbol is not None:
        # Single-symbol edge case
        codebook[node.symbol] = prefix if prefix != "" else "0"
        return codebook

    generate_codes(node.left, prefix + "0", codebook)
    generate_codes(node.right, prefix + "1", codebook)

    return codebook


# ---------------- ENCODE ----------------
def huffman_encode(data):
    """
    Encode data using Huffman coding
    Returns:
        encoded_bits
        tree
        codebook
    """
    if len(data) == 0:
        raise ValueError("Cannot Huffman encode empty data.")

    tree = build_huffman_tree(data)
    codebook = generate_codes(tree)

    encoded_bits = ''.join(codebook[x] for x in data)

    return encoded_bits, tree, codebook


# ---------------- SAVE ENCODED FILE ----------------
def save_huffman_encoded(data, filepath):
    """
    Save Huffman compressed binary file

    data = flattened cluster_map uint8 array
    filepath = output .bin file
    """
    if len(data) == 0:
        print("Warning: No data available for Huffman encoding.")
        return

    encoded_bits, tree, codebook = huffman_encode(data)

    # Safe padding
    padding = (8 - len(encoded_bits) % 8) % 8
    encoded_bits += "0" * padding

    # Convert bitstring to bytes
    byte_array = bytearray()
    for i in range(0, len(encoded_bits), 8):
        byte = encoded_bits[i:i+8]
        byte_array.append(int(byte, 2))

    # Save compressed file
    with open(filepath, "wb") as f:
        pickle.dump({
            "padding": padding,
            "tree": tree,
            "data": byte_array
        }, f)

    print(f"Huffman encoded file saved to: {filepath}")


# ---------------- LOAD ENCODED FILE ----------------
def load_huffman_encoded(filepath):
    """
    Load Huffman compressed file
    """
    with open(filepath, "rb") as f:
        obj = pickle.load(f)

    return obj["data"], obj["tree"], obj["padding"]


# ---------------- DECODE ----------------
def huffman_decode(byte_array, tree, padding):
    """
    Decode Huffman compressed data
    """
    bit_string = ""

    for byte in byte_array:
        bit_string += format(byte, '08b')

    # Remove padding bits safely
    if padding > 0:
        bit_string = bit_string[:-padding]

    decoded = []
    node = tree

    for bit in bit_string:
        if bit == "0":
            node = node.left
        else:
            node = node.right

        if node.symbol is not None:
            decoded.append(node.symbol)
            node = tree

    return decoded