#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>
#include <string>
#include <omp.h>

using namespace std;

// Node structure
struct Node
{
    char ch;
    int freq;
    Node *left, *right;

    Node(char c, int f)
    {
        ch = c;
        freq = f;
        left = right = nullptr;
    }
};

// Comparator for priority queue
struct compare
{
    bool operator()(Node *l, Node *r)
    {
        return l->freq > r->freq;
    }
};

// Build Huffman Tree (SEQUENTIAL)
Node *buildTree(const string &text)
{
    unordered_map<char, int> freq;

    for (char ch : text)
        freq[ch]++;

    priority_queue<Node *, vector<Node *>, compare> pq;

    for (auto pair : freq)
        pq.push(new Node(pair.first, pair.second));

    while (pq.size() > 1)
    {
        Node *left = pq.top();
        pq.pop();
        Node *right = pq.top();
        pq.pop();

        Node *merged = new Node('\0', left->freq + right->freq);
        merged->left = left;
        merged->right = right;

        pq.push(merged);
    }

    return pq.top();
}

// Generate Huffman Codes (SEQUENTIAL)
void generateCodes(Node *root, string code,
                   unordered_map<char, string> &codebook)
{
    if (!root)
        return;

    if (root->ch != '\0')
        codebook[root->ch] = code;

    generateCodes(root->left, code + "0", codebook);
    generateCodes(root->right, code + "1", codebook);
}

// Parallel Encoding (OPENMP)
string parallelEncode(const string &text,
                      unordered_map<char, string> &codebook)
{

    vector<string> encoded(text.size());

#pragma omp parallel for
    for (int i = 0; i < text.size(); i++)
    {
        encoded[i] = codebook[text[i]];
    }

    // Combine results
    string result = "";
    for (auto &s : encoded)
        result += s;

    return result;
}

// Sequential Encoding
string sequentialEncode(const string &text,
                        unordered_map<char, string> &codebook)
{
    string result = "";
    for (char ch : text)
        result += codebook[ch];

    return result;
}

// MAIN
int main()
{
    string text = "this is a huffman encoding test ";

    // Increase workload
    string largeText = "";
    for (int i = 0; i < 10000; i++)
        largeText += text;

    // Build tree
    Node *root = buildTree(largeText);

    unordered_map<char, string> codebook;
    generateCodes(root, "", codebook);

    // Sequential
    double start = omp_get_wtime();
    string seq = sequentialEncode(largeText, codebook);
    double end = omp_get_wtime();
    cout << "Sequential Time: " << (end - start) << " sec\n";

    // Parallel
    start = omp_get_wtime();
    string par = parallelEncode(largeText, codebook);
    end = omp_get_wtime();
    cout << "Parallel Time: " << (end - start) << " sec\n";

    cout << "Encoding correct: " << (seq == par) << endl;

    return 0;
}