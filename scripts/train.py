#!/usr/bin/env python3
"""
Contrastive Autoencoder v2 - Training Script
Run locally or on Kaggle/Colab
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import json
import os

# ===== CONFIG =====
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ===== DNA ENCODER =====
def encode_dna(sequences, window_size=1024):
    encoding = {"A": [1,0,0,0], "C": [0,1,0,0], "G": [0,0,1,0], "T": [0,0,0,1], "N": [0,0,0,0]}
    encoded = []
    for seq in sequences:
        seq = seq.upper()[:window_size].ljust(window_size, "N")
        encoded.append([encoding.get(b, [0,0,0,0]) for b in seq])
    return np.array(encoded, dtype=np.float32).reshape(-1, window_size * 4)

def decode_dna(encoded, window_size=1024):
    bases = ["A", "C", "G", "T", "N"]
    sequences = []
    for enc in encoded:
        enc = enc.reshape(window_size, 4)
        seq = "".join(bases[np.argmax(row)] if np.argmax(row) < 4 else "N" for row in enc)
        sequences.append(seq)
    return sequences

def gc_content(sequences):
    gc = []
    for seq in sequences:
        seq = seq.upper()
        g, c = seq.count("G"), seq.count("C")
        gc.append((g + c) / len(seq) * 100 if len(seq) > 0 else 0)
    return np.array(gc)

# ===== DATA GENERATOR =====
def generate_synthetic_genome(n_sequences=50000, window_size=1024, n_patterns=5, seed=42):
    np.random.seed(seed)
    patterns = []
    for i in range(n_patterns):
        gc = 0.2 + (i / (n_patterns - 1)) * 0.6
        freq = {"A": (1 - gc) / 2, "T": (1 - gc) / 2, "G": gc / 2, "C": gc / 2}
        for _ in range(n_sequences // n_patterns):
            seq = "".join(
                "A" if r < freq["A"] else
                "T" if r < freq["A"] + freq["T"] else
                "G" if r < freq["A"] + freq["T"] + freq["G"] else "C"
                for r in np.random.random(window_size)
            )
            patterns.append(seq)
    random.shuffle(patterns)
    return patterns[:n_sequences]

# ===== DATASETS =====
class GenomicDataset(Dataset):
    def __init__(self, encoded_sequences):
        self.data = torch.FloatTensor(encoded_sequences)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        if random.random() < 0.3:
            x = torch.clamp(x + torch.randn_like(x) * 0.05, 0, 1)
        return x, x

class ContrastiveDataset(Dataset):
    def __init__(self, encoded_sequences):
        self.data = torch.FloatTensor(encoded_sequences)
        from sklearn.metrics.pairwise import cosine_similarity
        print("Computing similarity matrix...")
        self.sim_matrix = cosine_similarity(encoded_sequences)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        sims = self.sim_matrix[idx]
        similar_idx = np.random.choice(np.where(sims > 0.9)[0] if len(np.where(sims > 0.9)[0]) > 0 else [idx])
        x_positive = self.data[similar_idx]
        dissimilar_idx = np.random.choice(np.where(sims < 0.3)[0] if len(np.where(sims < 0.3)[0]) > 0 else list(range(len(self.data))))
        x_negative = self.data[dissimilar_idx]
        return x, x_positive, x_negative

# ===== MODEL =====
class ContrastiveAutoencoder(nn.Module):
    def __init__(self, input_dim=4096, latent_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 2048), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2048, input_dim), nn.Sigmoid()
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

# ===== LOSS =====
class NT_XentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        z1, z2 = nn.functional.normalize(z1, dim=1), nn.functional.normalize(z2, dim=1)
        sim = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(len(z1), device=z1.device)
        return nn.functional.cross_entropy(sim, labels)

# ===== TRAINING =====
def train_epoch(model, loader, optimizer, criterion_mse, criterion_contrastive, lambda_contrastive=0.1):
    model.train()
    total_recon, total_contrastive = 0, 0
    for x, x_pos, x_neg in tqdm(loader, desc="Training"):
        x, x_pos = x.to("cuda"), x_pos.to("cuda")
        optimizer.zero_grad()
        x_recon, z = model(x)
        _, z_pos = model(x_pos)
        recon_loss = criterion_mse(x_recon, x)
        contrastive_loss = criterion_contrastive(z, z_pos)
        loss = recon_loss + lambda_contrastive * contrastive_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_recon += recon_loss.item()
        total_contrastive += contrastive_loss.item()
    return total_recon / len(loader), total_contrastive / len(loader)

def validate(model, loader, criterion_mse):
    model.eval()
    total_recon = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to("cuda")
            x_recon, _ = model(x)
            total_recon += criterion_mse(x_recon, x).item()
    return total_recon / len(loader)

# ===== MAIN =====
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate data
    print("Generating synthetic data...")
    synthetic_seqs = generate_synthetic_genome(args.n_sequences, args.window_size, args.n_patterns)
    X = encode_dna(synthetic_seqs, args.window_size)
    
    # Split
    train_size = int(0.9 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    
    # Datasets
    train_dataset = ContrastiveDataset(X_train)
    val_dataset = GenomicDataset(X_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Model
    model = ContrastiveAutoencoder(args.window_size * 4, args.latent_dim).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    criterion_mse = nn.MSELoss()
    criterion_contrastive = NT_XentLoss(temperature=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    history = {"train_recon": [], "train_contrastive": [], "val_recon": []}
    
    for epoch in range(args.epochs):
        train_recon, train_contrast = train_epoch(model, train_loader, optimizer, criterion_mse, criterion_contrastive, args.lambda_contrastive)
        val_recon = validate(model, val_loader, criterion_mse)
        scheduler.step()
        
        history["train_recon"].append(train_recon)
        history["train_contrastive"].append(train_contrast)
        history["val_recon"].append(val_recon)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Recon: {train_recon:.4f} | Contrastive: {train_contrast:.4f} | Val: {val_recon:.4f}")
        
        if val_recon < getattr(history, "best_val", float('inf')):
            history["best_val"] = val_recon
            torch.save(model.state_dict(), "best_model.pth")
    
    # Save history
    with open("history.json", "w") as f:
        json.dump(history, f)
    
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sequences", type=int, default=50000)
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--n_patterns", type=int, default=5)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_contrastive", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
