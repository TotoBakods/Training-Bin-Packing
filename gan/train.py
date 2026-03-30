import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from data_loader import get_dataloaders
from model import Generator, Discriminator
import pickle

# Parameters
LATENT_DIM = 100
EPOCHS = 500
BATCH_SIZE = 64
LR = 0.0002
B1 = 0.5
B2 = 0.999

# Get absolute path to datasets.csv assuming it is in the parent directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(current_dir, '..', 'datasets', 'datasets.csv')
CHECKPOINT_DIR = os.path.join(current_dir, 'checkpoints')
SCALER_PATH = os.path.join(current_dir, 'scaler.pkl')
LOSS_HISTORY_PATH = os.path.join(current_dir, 'loss_history.json')

def train():
    # Setup directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    print(f"Loading data from {DATA_FILE}...")
    train_loader, val_loader, scaler = get_dataloaders(DATA_FILE, BATCH_SIZE)
    
    # Save scaler for generation
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
        
    # Model
    generator = Generator(LATENT_DIM, 4).to(device)
    discriminator = Discriminator(4).to(device)
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(B1, B2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(B1, B2))
    
    # Loss
    adversarial_loss = nn.BCELoss()
    
    history = {"d_loss": [], "g_loss": [], "val_d_loss": [], "val_g_loss": [], "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR}
    
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        epoch_d_loss = 0
        epoch_g_loss = 0
        batch_count = 0
        
        for i, imgs in enumerate(train_loader):
            
            # Configure input
            real_imgs = imgs.to(device)
            batch_size = real_imgs.size(0)
            
            valid = torch.ones(batch_size, 1, requires_grad=False).to(device)
            fake = torch.zeros(batch_size, 1, requires_grad=False).to(device)
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, LATENT_DIM).to(device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            batch_count += 1

            # Log progress
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(train_loader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
        
        # Save average epoch loss
        history["d_loss"].append(epoch_d_loss / max(1, batch_count))
        history["g_loss"].append(epoch_g_loss / max(1, batch_count))

        # --- Validation Loop ---
        generator.eval()
        discriminator.eval()
        val_d_loss, val_g_loss = 0, 0
        val_batches = 0
        with torch.no_grad():
            for v_imgs in val_loader:
                real_imgs = v_imgs.to(device)
                b_size = real_imgs.size(0)
                
                valid = torch.ones(b_size, 1).to(device)
                fake = torch.zeros(b_size, 1).to(device)
                
                # Gen loss
                z = torch.randn(b_size, LATENT_DIM).to(device)
                gen_imgs = generator(z)
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)
                val_g_loss += g_loss.item()
                
                # Disc loss
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                val_d_loss += d_loss.item()
                
                val_batches += 1
                
        history["val_d_loss"].append(val_d_loss / max(1, val_batches))
        history["val_g_loss"].append(val_g_loss / max(1, val_batches))
        
        generator.train()
        discriminator.train()

        # Save checkpoint periodically
        if epoch % 5 == 0:
             torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, "generator.pth"))
             torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, "discriminator.pth"))

    # Final save
    torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, "generator.pth"))
    
    # Save loss history
    import json
    with open(LOSS_HISTORY_PATH, 'w') as f:
        json.dump(history, f)
        
    print(f"Training finished. Model saved. Loss history saved to {LOSS_HISTORY_PATH}")

if __name__ == "__main__":
    train()
