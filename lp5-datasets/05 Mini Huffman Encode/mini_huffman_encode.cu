// Mini Project: Implement Huffman Encoding on GPU

#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>
#include <string>
#include <cuda.h>

using namespace std;

// Huffman Tree Node
struct Node {
    char ch;
    int freq;
    Node *left, *right;

    Node(char c, int f) {
        ch = c;
        freq = f;
        left = right = NULL;
    }
};

// Comparator for priority queue
struct compare {
    bool operator()(Node* l, Node* r) {
        return l->freq > r->freq;
    }
};

// Build Huffman Tree (CPU)
Node* buildTree(unordered_map<char, int>& freq) {
    priority_queue<Node*, vector<Node*>, compare> pq;

    for (auto pair : freq) {
        pq.push(new Node(pair.first, pair.second));
    }

    while (pq.size() != 1) {
        Node* left = pq.top(); pq.pop();
        Node* right = pq.top(); pq.pop();

        Node* sum = new Node('\0', left->freq + right->freq);
        sum->left = left;
        sum->right = right;

        pq.push(sum);
    }

    return pq.top();
}

// Generate Codes (CPU)
void generateCodes(Node* root, string code,
                   unordered_map<char, string>& huffmanCode) {
    if (!root) return;

    if (!root->left && !root->right) {
        huffmanCode[root->ch] = code;
    }

    generateCodes(root->left, code + "0", huffmanCode);
    generateCodes(root->right, code + "1", huffmanCode);
}

// GPU Kernel: Frequency Count
__global__ void countFrequency(char* text, int* freq, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        atomicAdd(&freq[(int)text[idx]], 1);
    }
}

// GPU Kernel: Encoding
__global__ void encodeText(char* text, int n,
                           char* d_codes,
                           int* d_offsets,
                           char* output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        char c = text[idx];
        int offset = d_offsets[(int)c];

        for (int i = 0; d_codes[offset + i] != '\0'; i++) {
            output[idx * 32 + i] = d_codes[offset + i];
        }
    }
}

// Main
int main() {
    string text = "parallel huffman encoding using gpu";

    int n = text.size();


    // GPU Frequency Count

    char* d_text;
    int* d_freq;

    cudaMalloc(&d_text, n * sizeof(char));
    cudaMalloc(&d_freq, 256 * sizeof(int));

    cudaMemcpy(d_text, text.c_str(), n, cudaMemcpyHostToDevice);
    cudaMemset(d_freq, 0, 256 * sizeof(int));

    countFrequency<<<(n + 255)/256, 256>>>(d_text, d_freq, n);

    int h_freq[256];
    cudaMemcpy(h_freq, d_freq, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    unordered_map<char, int> freq;
    for (int i = 0; i < 256; i++) {
        if (h_freq[i] > 0)
            freq[(char)i] = h_freq[i];
    }


    // Build Tree (CPU)

    Node* root = buildTree(freq);

    unordered_map<char, string> huffmanCode;
    generateCodes(root, "", huffmanCode);

    cout << "Huffman Codes:\n";
    for (auto pair : huffmanCode) {
        cout << pair.first << ": " << pair.second << endl;
    }


    // Prepare codes for GPU

    vector<char> codes;
    int offsets[256] = {0};

    for (int i = 0; i < 256; i++) {
        char c = (char)i;
        if (huffmanCode.find(c) != huffmanCode.end()) {
            offsets[i] = codes.size();
            for (char bit : huffmanCode[c]) {
                codes.push_back(bit);
            }
            codes.push_back('\0');
        }
    }

    char* d_codes;
    int* d_offsets;
    char* d_output;

    cudaMalloc(&d_codes, codes.size());
    cudaMalloc(&d_offsets, 256 * sizeof(int));
    cudaMalloc(&d_output, n * 32 * sizeof(char));

    cudaMemcpy(d_codes, codes.data(), codes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets, 256 * sizeof(int), cudaMemcpyHostToDevice);


    // GPU Encoding

    encodeText<<<(n + 255)/256, 256>>>(d_text, n,
                                      d_codes, d_offsets,
                                      d_output);

    char* h_output = new char[n * 32];
    cudaMemcpy(h_output, d_output, n * 32, cudaMemcpyDeviceToHost);

    cout << "\nEncoded Output:\n";
    for (int i = 0; i < n; i++) {
        cout << &h_output[i * 32] << " ";
    }
    cout << endl;


    // Cleanup

    cudaFree(d_text);
    cudaFree(d_freq);
    cudaFree(d_codes);
    cudaFree(d_offsets);
    cudaFree(d_output);

    return 0;
}

// COMMAND:
// nvcc mini_huffman_encode.cu -o mini_huffman_encode
// ./mini_huffman_encode

// CODE RUNS ON COLLAB
// !nvcc mini_huffman_encode.cu -o mini_huffman_encode && ./mini_huffman_encode