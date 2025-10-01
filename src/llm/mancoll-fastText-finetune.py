import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

import random
import numpy as np

def reset_seed(seed=42):
    """Reset all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# ===============================
# 1. Load data
# ===============================
for i in {5, 10, 15, 20, 30, 40}:  # 5%, 10%, 15%, 20%, 30%, 40%
# for i in range(200,601,200):
    reset_seed(42)
    file_path = f"data/processed_data/case_info_2021_{i}perc_noise.xlsx"
    print(f"Training with {i} samples")
    print(f"Loading data from {file_path}")

    df = pd.read_excel(file_path)
    test_path = "data/processed_data/case_info_2020.xlsx"

    texts = df["SUMMARY"].astype(str).tolist()
    labels = df["MANCOLLNEW"].astype(int).tolist()

    unique_labels = sorted(set(labels))
    label2id = {v: i for i, v in enumerate(unique_labels)}
    labels = [label2id[l] for l in labels]
    num_classes = len(unique_labels)
    print(f"Number of classes: {num_classes}, label2id: {label2id}")

    # Split into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42
    )

    # ===============================
    # 2. Build vocabulary
    # ===============================
    def tokenize(text):
        return text.lower().split()

    counter = Counter()
    for text in train_texts:
        counter.update(tokenize(text))

    vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common(20000))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    def encode(text):
        return [vocab.get(word, vocab["<UNK>"]) for word in tokenize(text)]

    # ===============================
    # 3. Dataset & DataLoader
    # ===============================
    class CrashDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = [torch.tensor(encode(t), dtype=torch.long) for t in texts]
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.texts[idx], self.labels[idx]

    def collate_fn(batch):
        texts, labels = zip(*batch)
        texts_pad = pad_sequence(texts, batch_first=True, padding_value=0)
        return texts_pad, torch.tensor(labels)

    train_dataset = CrashDataset(train_texts, train_labels)
    val_dataset = CrashDataset(val_texts, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # ===============================
    # 4. Define FastText model
    # ===============================
    class FastText(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_classes):
            super(FastText, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.fc = nn.Linear(embed_dim, num_classes)

        def forward(self, x):
            emb = self.embedding(x)       # [batch, seq_len, embed_dim]
            out = emb.mean(dim=1)         # Average pooling
            return self.fc(out)           # [batch, num_classes]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastText(vocab_size=len(vocab), embed_dim=128, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ===============================
    # 5. Training
    # ===============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} finished.")

    # ===============================
    # 6. Validation evaluation
    # ===============================
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"[Val] Accuracy: {acc:.4f}")
    print(f"[Val] Macro F1: {macro_f1:.4f}")

    # ===============================
    # 7. Test set evaluation
    # ===============================
    df_test = pd.read_excel(test_path)
    test_texts = df_test["SUMMARY"].astype(str).tolist()
    test_labels = df_test["MANCOLL"].astype(int).tolist()

    # Map test labels to training label IDs; skip any unseen labels
    test_labels = [label2id.get(l, -1) for l in test_labels]
    test_pairs = [(t, l) for t, l in zip(test_texts, test_labels) if l != -1]
    test_texts, test_labels = zip(*test_pairs)

    test_dataset = CrashDataset(test_texts, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            preds = outputs.argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(batch_y.cpu().numpy())

    test_acc = accuracy_score(test_true, test_preds)
    test_macro_f1 = f1_score(test_true, test_preds, average="macro")

    print(f"[Test] Accuracy: {test_acc:.4f}")
    print(f"[Test] Macro F1: {test_macro_f1:.4f}")

    # Evaluate again excluding the last class (often "Unknown")
    last_class = 6  # Assume the last class ID is 6
    mask = [y != last_class for y in test_labels]

    filtered_true = [y for y, m in zip(test_true, mask) if m]
    filtered_preds = [y for y, m in zip(test_preds, mask) if m]

    test_acc_excl = accuracy_score(filtered_true, filtered_preds)
    test_macro_f1_excl = f1_score(filtered_true, filtered_preds, average="macro")

    print(f"[Test excl. last class={last_class}] Accuracy: {test_acc_excl:.4f}")
    print(f"[Test excl. last class={last_class}] Macro F1: {test_macro_f1_excl:.4f}")
