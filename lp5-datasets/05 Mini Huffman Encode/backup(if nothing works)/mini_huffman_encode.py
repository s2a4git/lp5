# Mini Project: Implement Huffman Encoding on GPU

import heapq
from collections import Counter
from multiprocessing import Pool
import time


# Node Class
class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


# Build Huffman Tree
def build_tree(text):
    freq = Counter(text)
    heap = [Node(char, freq) for char, freq in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right

        heapq.heappush(heap, merged)

    return heap[0]


# Generate Codes
def generate_codes(node, prefix="", codebook={}):
    if node is None:
        return

    if node.char is not None:
        codebook[node.char] = prefix

    generate_codes(node.left, prefix + "0", codebook)
    generate_codes(node.right, prefix + "1", codebook)

    return codebook


# Encode Chunk (Parallel)
def encode_chunk(args):
    chunk, codebook = args
    return "".join([codebook[ch] for ch in chunk])


# Parallel Encoding
def parallel_encode(text, codebook, num_workers=4):
    chunk_size = len(text) // num_workers
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    with Pool(num_workers) as p:
        result = p.map(encode_chunk, [(chunk, codebook) for chunk in chunks])

    return "".join(result)


# Sequential Encoding
def sequential_encode(text, codebook):
    return "".join([codebook[ch] for ch in text])


# MAIN
if __name__ == "__main__":
    text = "this is a huffman encoding test " * 10000

    # Build tree
    root = build_tree(text)
    codebook = generate_codes(root)

    # Sequential
    start = time.time()
    seq_encoded = sequential_encode(text, codebook)
    print("Sequential Time:", time.time() - start)

    # Parallel
    start = time.time()
    par_encoded = parallel_encode(text, codebook)
    print("Parallel Time:", time.time() - start)

    print("Encoding successful:", seq_encoded == par_encoded)
