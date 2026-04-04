import torch
import torch.nn as nn
import torch.optim as optim
import os
from data_loader import get_gpu_dataset
from model import Generator, Discriminator
import pickle

# Parameters
LATENT_DIM = 100
EPOCHS = 1000
BATCH_SIZE = 256
LR_G = 0.0002
LR_D = 0.0002
B1 = 0.5
B2 = 0.999
CRITIC_ITERATIONS = 1  # Standard GAN treats them 1:1 usually
TARGET_LOSS = 0.693    # -ln(0.5) is the ideal for BCE
EARLY_STOP_PARITY_THRESHOLD = 0.0
EARLY_STOP_PATIENCE = 100

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
    if device.type == 'cuda':
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = False
    print(f"Using device: {device}")
    
    history = {
        "parameters": {
            "latent_dim": LATENT_DIM,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr_g": LR_G,
            "lr_d": LR_D,
            "betas": [B1, B2],
            "target_loss": TARGET_LOSS,
            "early_stop_parity_threshold": EARLY_STOP_PARITY_THRESHOLD,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "device": str(device)
        },
        "d_loss": [], "g_loss": [], "val_d_loss": [], "val_g_loss": [],
        "parity": [], "lr_g_history": [], "lr_d_history": [],
        "dte_d": [], "dte_g": [],
        "convergence_epoch": None, "convergence_reason": None
    }
    
    print("--- Training Parameters ---")
    for k, v in history["parameters"].items():
        print(f"{k}: {v}")
    print("---------------------------")

    # Data
    print(f"Loading data from {DATA_FILE}...")
    full_data, scaler = get_gpu_dataset(DATA_FILE, device)
    
    # Simple split
    n_total = full_data.size(0)
    indices = torch.randperm(n_total)
    n_val = int(n_total * 0.2)
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    train_data = full_data[train_idx]
    val_data = full_data[val_idx]
    
    n_train = train_data.size(0)
    n_batches = (n_train + BATCH_SIZE - 1) // BATCH_SIZE

    # Save scaler for generation
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
        
    # Model
    generator = Generator(LATENT_DIM, 4).to(device)
    discriminator = Discriminator(4).to(device)
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=LR_G, betas=(B1, B2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(B1, B2))
    
    # LR Schedulers
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=EPOCHS, eta_min=1e-5)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=EPOCHS, eta_min=1e-5)

    adversarial_loss = nn.BCELoss()
    
    # Create DataLoader instead of manual indexing on GPU
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = TensorDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    import time
    start_time = time.time()

    for epoch in range(EPOCHS):
        generator.train()
        discriminator.train()
        
        epoch_d_loss = 0
        epoch_g_loss = 0
        batch_count = 0
        
        for batch_idx, (real_imgs,) in enumerate(train_loader):
            current_batch_size = real_imgs.size(0)
            batch_count += 1
            
            # Ground truth labels with one-sided smoothing
            valid = torch.full((current_batch_size, 1), 0.9, device=device) # Label Smoothing
            fake_labels = torch.zeros((current_batch_size, 1), device=device)
            
            # --- Train Generator ---
            optimizer_G.zero_grad()
            z = torch.randn(current_batch_size, LATENT_DIM, device=device)
            gen_imgs = generator(z)
            
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()
            epoch_g_loss += g_loss.item()
            
            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            
            # Instance Noise
            noise = torch.randn_like(real_imgs) * 0.01
            real_noisy = real_imgs + noise
            
            real_loss = adversarial_loss(discriminator(real_noisy), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake_labels)
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            epoch_d_loss += d_loss.item()
        
        # Scheduler updates
        scheduler_G.step()
        scheduler_D.step()
        
        # History
        history["d_loss"].append(epoch_d_loss / batch_count)
        history["g_loss"].append(epoch_g_loss / batch_count)
        
        # Validation
        val_d_loss = 0
        val_g_loss = 0
        with torch.no_grad():
            generator.eval()
            discriminator.eval()
            for val_idx, (real_imgs_val,) in enumerate(val_loader):
                z = torch.randn(real_imgs_val.size(0), LATENT_DIM, device=device)
                gen_imgs_val = generator(z)
                
                v_labels = torch.ones((real_imgs_val.size(0), 1), device=device)
                f_labels = torch.zeros((real_imgs_val.size(0), 1), device=device)
                
                v_loss = adversarial_loss(discriminator(real_imgs_val), v_labels)
                f_loss = adversarial_loss(discriminator(gen_imgs_val), f_labels)
                
                val_d_loss += (v_loss.item() + f_loss.item()) / 2
                val_g_loss += adversarial_loss(discriminator(gen_imgs_val), v_labels).item()
        
        n_val_batches = len(val_loader)
        history["val_d_loss"].append(val_d_loss / n_val_batches)
        history["val_g_loss"].append(val_g_loss / n_val_batches)
        
        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1}/{EPOCHS}] D_Loss: {history['d_loss'][-1]:.4f} G_Loss: {history['g_loss'][-1]:.4f}")
            torch.cuda.empty_cache()
        else:
            # Carry over previous validation loss for reporting
            history["val_d_loss"].append(history["val_d_loss"][-1] if history["val_d_loss"] else 0)
            history["val_g_loss"].append(history["val_g_loss"][-1] if history["val_g_loss"] else 0)

        # Parity tracking + LR scheduler step
        epoch_parity = abs(history["d_loss"][-1] - history["g_loss"][-1])
        history["parity"].append(epoch_parity)
        
        dte_d = abs(history["d_loss"][-1] - TARGET_LOSS)
        dte_g = abs(history["g_loss"][-1] - TARGET_LOSS)
        history["dte_d"].append(dte_d)
        history["dte_g"].append(dte_g)
        
        history["lr_g_history"].append(optimizer_G.param_groups[0]['lr'])
        history["lr_d_history"].append(optimizer_D.param_groups[0]['lr'])
        
        print(f"[Epoch {epoch}/{EPOCHS}] [D Loss: {history['d_loss'][-1]:.5f}] [G Loss: {history['g_loss'][-1]:.5f}]")
        print(f"      [Parity: {epoch_parity:.5f}] [DTE-D: {dte_d:.5f}] [DTE-G: {dte_g:.5f}]")
        
        scheduler_G.step()
        scheduler_D.step()

        generator.train()
        discriminator.train()

        # Save checkpoint periodically
        if epoch % 5 == 0:
            torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, "generator.pth"))
            torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, "discriminator.pth"))

        # Save best-parity checkpoint
        if history["parity"][-1] == min(history["parity"]):
            torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, "generator_best_parity.pth"))
            torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, "discriminator_best_parity.pth"))

        # Save loss history periodically
        if epoch % 20 == 0:
            import json
            with open(LOSS_HISTORY_PATH, 'w') as f:
                json.dump(history, f, indent=4)

        # Early stopping: halt when Nash equilibrium parity is stable
        if len(history["parity"]) >= EARLY_STOP_PATIENCE:
            if all(p < EARLY_STOP_PARITY_THRESHOLD for p in history["parity"][-EARLY_STOP_PATIENCE:]):
                history["convergence_epoch"] = epoch
                history["convergence_reason"] = (
                    f"Nash equilibrium: parity < {EARLY_STOP_PARITY_THRESHOLD} "
                    f"for {EARLY_STOP_PATIENCE} consecutive epochs"
                )
                print(f"[Early Stop] Epoch {epoch}: {history['convergence_reason']}")
                break

    # Final save
    torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, "discriminator.pth"))
    
    total_time = time.time() - start_time
    history["total_training_time_seconds"] = total_time
    history["final_d_loss"] = history["d_loss"][-1]
    history["final_g_loss"] = history["g_loss"][-1]
    history["final_parity"] = history["parity"][-1]
    
    # Save loss history
    import json
    with open(LOSS_HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=4)
        
    print(f"Training finished in {total_time:.2f} seconds.")
    print(f"Final Parity: {history['final_parity']:.6f}")
    print(f"Model saved. Loss history saved to {LOSS_HISTORY_PATH}")

if __name__ == "__main__":
    train()
