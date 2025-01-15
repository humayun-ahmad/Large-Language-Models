import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
import re

# Ensure you have the NLTK tokenizer data
nltk.download('punkt')

# Load and preprocess the text
def preprocess_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()
        text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
        words = word_tokenize(text)
    return words

# Create context-target pairs
def create_context_target_pairs(words, window_size):
    pairs = []
    for i in range(window_size, len(words) - window_size):
        context = words[i - window_size:i] + words[i + 1:i + window_size + 1]
        target = words[i]
        pairs.append((context, target))
    return pairs

# Encode words to integers
def build_vocab(words):
    vocab = {word: idx for idx, word in enumerate(set(words))}
    reverse_vocab = {idx: word for word, idx in vocab.items()}
    return vocab, reverse_vocab

# Convert context-target pairs to indices
def convert_to_indices(pairs, vocab):
    indexed_pairs = []
    for context, target in pairs:
        context_indices = [vocab[word] for word in context]
        target_index = vocab[target]
        indexed_pairs.append((context_indices, target_index))
    return indexed_pairs

# Define the CBOW model
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        embedded = self.embeddings(context)  # (batch_size, context_size, embedding_dim)
        context_embedding = embedded.mean(dim=1)  # Average embeddings across context words
        output = self.linear(context_embedding)  # Predict target word
        return output

# Train the CBOW model
def train_cbow_model(pairs, vocab_size, embedding_dim, epochs=5, learning_rate=0.01, batch_size=64):
    model = CBOW(vocab_size, embedding_dim)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            context_batch = torch.tensor([x[0] for x in batch], dtype=torch.long)
            target_batch = torch.tensor([x[1] for x in batch], dtype=torch.long)

            model.zero_grad()
            predictions = model(context_batch)
            loss = loss_function(predictions, target_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    return model

# Save embeddings to visualize or reuse
def save_embeddings(model, reverse_vocab, file_name="embeddings.txt"):
    with open(file_name, 'w') as f:
        for idx, word in reverse_vocab.items():
            embedding = model.embeddings.weight[idx].detach().numpy()
            embedding_str = ' '.join(map(str, embedding))
            f.write(f"{word} {embedding_str}\n")

# Main execution
if __name__ == "__main__":
    # Load and preprocess text
    file_path = "sherlock_holmes.txt"  # Replace with your file path
    words = preprocess_text(file_path)

    # Hyperparameters
    window_size = 2
    embedding_dim = 100
    epochs = 10
    batch_size = 64

    # Prepare data
    vocab, reverse_vocab = build_vocab(words)
    pairs = create_context_target_pairs(words, window_size)
    indexed_pairs = convert_to_indices(pairs, vocab)

    # Train the model
    model = train_cbow_model(indexed_pairs, len(vocab), embedding_dim, epochs, batch_size=batch_size)

    # Save embeddings
    save_embeddings(model, reverse_vocab)
    print("Embeddings saved to 'embeddings.txt'.")
