import sys
import argparse
import json
import re
import time
from datetime import datetime, timezone
import torch
import torch.nn as nn

VOCAB_SIZE = 3500  # initial target, actual vocab_size depends on count of unique words
HIDDEN_DIM = 256
EMBEDDING_DIM = 64
PARAM_LIMIT = 2000000

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    return args.input, args.output, args.epochs, args.batch_size

def load_data(input):
    print("Loading abstracts from papers.json...", flush=True)
    try:
        with open(input, 'r') as file:
            data = json.load(file)
        arxiv_id = []
        abstract = []
        for paper in data:
            if "abstract" in paper and "arxiv_id" in paper:
                arxiv_id.append(paper["arxiv_id"])
                abstract.append(paper["abstract"])
    except FileNotFoundError:
        print(f'Error: file not found: {input}')
        exit(1)
    except Exception as e:
        print(f'Error: {str(e)}')
        exit(1)
    print(f'Found {len(abstract)} abstracts', flush=True)
    return arxiv_id, abstract

def clean_text(text):
    # Remove non-alphabetic characters and convert to lowercase
    cleaned_text = re.sub(r'[^a-zA-Z ]', '', text).lower()
    # Split into words and remove very short words (< 2 char)
    words = [w for w in cleaned_text.split() if len(w) >= 2]
    return words

def build_vocab(words):
    # Extract unique words with their frequency
    word_freq = {}
    for w in words:
        for word in w:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    # Keep top-k most frequent and create word-to-index mapping
    vocab_to_idx = {}
    idx_to_vocab = {}
    for i in range(VOCAB_SIZE):
        if len(word_freq) == 0: break
        top_word = max(word_freq, key=lambda k: word_freq[k])
        word_freq.pop(top_word)
        vocab_to_idx[top_word] = i+1 # reserve 0 for unknown
        idx_to_vocab[str(i+1)] = top_word
    return vocab_to_idx, idx_to_vocab

def encode_seq(words, vocab_to_idx, fixed_len):
    BoW = []
    for w in words:
        # pad or truncate to fixed length
        if len(w) >= fixed_len:
            abs = w[:fixed_len]  # truncate
        else:
            abs = w + [""] * (fixed_len - len(w))  # pad
        # Convert abstract to sequence of indices
        abs_idx = []
        for word in abs:
            if word in vocab_to_idx:
                abs_idx.append(vocab_to_idx[word])
            else:
                abs_idx.append(0)
        # Create bag_of_words representation
        bag_of_words = []
        for v in vocab_to_idx:
            val = 1 if abs_idx.count(vocab_to_idx[v]) > 0 else 0
            bag_of_words.append(val)
        BoW.append(bag_of_words)
    return BoW

def time_stamp():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

class TextAutoencoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_dim):
        super().__init__()
        # Encoder: vocab_size → hidden_dim → embedding_dim
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Decoder: embedding_dim → hidden_dim → vocab_size  
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
            nn.Sigmoid()  # Output probabilities
        )
    
    def forward(self, x):
        # Encode to bottleneck
        embedding = self.encoder(x)
        # Decode back to vocabulary space
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

def create_batches(data, batch_size):
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i: i+batch_size])
    return batches

def format_f(val, precision):
    return float(f'{val:.{precision}f}')

def train_model(epochs, batches, vocab_size, hidden_dim, embedding_dim):
    print("Training autoencoder...", flush=True)
    model = TextAutoencoder(vocab_size, hidden_dim, embedding_dim)
    learning_rate = 0.01  # best(smallest loss) in my experiment
    loss_func = nn.BCELoss()  # Binary cross-entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_time = time.time()

    # Training loop
    for i in range(epochs):
        for batch in batches:
            input = torch.tensor(batch, dtype=torch.float)
            reconstruction, embedding = model(input)
            loss = loss_func(reconstruction, input)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {i+1}/{epochs}, Loss: {format_f(loss.item(), 4)}', flush=True)

    print(f'Training complete in {format_f(time.time()-start_time, 2)} seconds', flush=True)
    return model, format_f(loss.item(), 4)

def create_embedding(BoW, model):
    loss_func = nn.BCELoss()
    input = torch.tensor(BoW, dtype=torch.float)
    reconstruction, embedding = model(input)
    loss = loss_func(reconstruction, input)
    return [format_f(em, 4) for em in embedding], format_f(loss.item(), 4)

def main():
    input, output, epochs, batch_size = parse_arg()
    output = "/data/output"

    # Load data
    arxiv_id, abstract = load_data(input)

    # Clean text
    words = []
    total_words = 0
    for text in abstract:
        w = clean_text(text)
        words.append(w)
        total_words += len(w)
    print(f'Building vocabulary from {total_words} words...', flush=True)

    # Build vocabulary
    vocab_to_idx, idx_to_vocab = build_vocab(words)
    vocab_size = len(vocab_to_idx)  # incase count of unique words < VOCAB_SIZE
    print(f'Vocabulary size: {vocab_size} words', flush=True)

    # Encode Sequence
    fixed_len = int(total_words/len(words))
    BoW = encode_seq(words, vocab_to_idx, fixed_len)

    # Calculate parameters
    print(f'Model architecture: {vocab_size} → {HIDDEN_DIM} → {EMBEDDING_DIM} → {HIDDEN_DIM} → {vocab_size}', flush=True)
    input_param = vocab_size * HIDDEN_DIM + HIDDEN_DIM
    encod_param = HIDDEN_DIM * EMBEDDING_DIM + EMBEDDING_DIM
    decod_param = EMBEDDING_DIM * HIDDEN_DIM + HIDDEN_DIM
    output_param = HIDDEN_DIM * vocab_size + vocab_size
    total_param = input_param + encod_param + decod_param + output_param
    under_limit = "✓" if total_param <= PARAM_LIMIT else "✗"
    print(f'Total parameters: {total_param} (under 2,000,000 limit {under_limit})', flush=True)

    # Start training
    start_time = time_stamp()
    batches = create_batches(BoW, batch_size)
    model, final_loss = train_model(epochs, batches, vocab_size, HIDDEN_DIM, EMBEDDING_DIM)
    end_time = time_stamp()

    # save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_to_idx': vocab_to_idx,
        'model_config': {
            'vocab_size': vocab_size,
            'hidden_dim': HIDDEN_DIM, 
            'embedding_dim': EMBEDDING_DIM
        }
    }, f'{output}/model.pth')

    # load model (for testing)
    #model2 = TextAutoencoder(vocab_size, HIDDEN_DIM, EMBEDDING_DIM)
    #model2.load_state_dict(torch.load(f'{output}/model.pth')["model_state_dict"])
    #model2.eval()

    # Generate output
    embeddings = []
    for i in range(len(BoW)):
        embedding, loss = create_embedding(BoW[i], model)
        embeddings.append({
            "arxiv_id": arxiv_id[i],
            "embedding": embedding,
            "reconstruction_loss": loss
        })
    with open(f'{output}/embeddings.json', 'w') as file:
        json.dump(embeddings, file, indent=2)

    vocabulary = {
        "vocab_to_idx": vocab_to_idx,
        "idx_to_vocab": idx_to_vocab,
        "vocab_size": vocab_size,
        "total_words": total_words
    }
    with open(f'{output}/vocabulary.json', 'w') as file:
        json.dump(vocabulary, file, indent=2)

    training_log = {
        "start_time": start_time,
        "end_time": end_time,
        "epochs": epochs,
        "final_loss": final_loss,
        "total_parameters": total_param,
        "papers_processed": len(abstract),
        "embedding_dimension": EMBEDDING_DIM
    }
    with open(f'{output}/training_log.json', 'w') as file:
        json.dump(training_log, file, indent=2)

if __name__ == "__main__":
    main()