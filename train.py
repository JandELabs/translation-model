import torch
import torch.nn as nn
from models.embeddings import (
    Transformer,
    Vocabulary,
    load_parallel_data,
    build_vocabularies,
    create_dataloaders
)

def create_loss_function():
    """
    Create Cross-Entropy loss that ignores padding.
    """
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    return criterion

def create_optimizer(model, learning_rate=0.0001):
    """
    Create Adam optimizer for the model.
    
    model: The Transformer model
    learning_rate: How big the weight updates are (default: 0.0001)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer

def setup_device():
    """
    Check if GPU is available and return the device.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU (training will be slower)")
    
    return device

def train_step(model, batch, criterion, optimizer, device):
    """
    Perform one training step on a batch.
    
    model: The Transformer
    batch: Dictionary with 'src', 'tgt_input', 'tgt_output'
    criterion: Loss function
    optimizer: Optimizer
    device: CPU or GPU
    
    Returns: loss value
    """
    model.train()  # Set model to training mode

      # Get data from batch and move to device
    src = batch['src'].to(device)           # [batch_size, src_seq_len]
    tgt_input = batch['tgt_input'].to(device)   # [batch_size, tgt_seq_len]
    tgt_output = batch['tgt_output'].to(device) # [batch_size, tgt_seq_len]
    logits = model(src, tgt_input)
    vocab_size = logits.size(-1)
    logits_flat = logits.view(-1, vocab_size) # [batch*seq_len, vocab_size]
    tgt_output_flat = tgt_output.view(-1)
    # Calculate loss
    loss = criterion(logits_flat, tgt_output_flat)
    # Backward pass - calculate gradients
    optimizer.zero_grad()   # Clear old gradients
    loss.backward()         # Calculate new gradients
    optimizer.step()

    # Return loss value (detach from computation graph)
    return loss.item()


def validation_step(model, batch, criterion, device):
    """
    Perform one validation step on a batch.
    """
    model.eval()
    
    # Disable gradient computation (saves memory and speeds up)
    with torch.no_grad():
        # Get data and move to device
        src = batch['src'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)

        # Forward pass
        logits = model(src, tgt_input)
        
        # Reshape for loss
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)
        tgt_output_flat = tgt_output.view(-1)
        
        # Calculate loss
        loss = criterion(logits_flat, tgt_output_flat)
        
        return loss.item()

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one complete epoch.
    
    Returns: average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    # Loop through all batches
    for batch in train_loader:
        # Train on this batch
        loss = train_step(model, batch, criterion, optimizer, device)
        
        # Accumulate loss
        total_loss += loss
        num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss
    
def validate_epoch(model, val_loader, criterion, device):
    """
    Validate for one complete epoch.
    
    Returns: average validation loss
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # Loop through all validation batches
    for batch in val_loader:
        # Validate on this batch
        loss = validation_step(model, batch, criterion, device)
        
        # Accumulate loss
        total_loss += loss
        num_batches += 1
    
    # Calculate average loss
    avg_loss = total_loss / num_batches
    return avg_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    """
    Complete training loop.
    """
    best_val_loss = float('inf')
    
    print("Starting training...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate for one epoch
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  → New best model saved! (Val Loss: {val_loss:.4f})")
    
    print("-" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    return best_val_loss

def main():
    """
    Main function to train the translation model.
    """
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.0001
    D_MODEL = 512
    NUM_HEADS = 8
    D_FF = 2048
    NUM_LAYERS = 6
    DROPOUT = 0.1

    # Load data
    print("Loading parallel data...")
    data_file = 'data/parallel_data.txt'
    pairs = load_parallel_data(data_file)
    print(f"Loaded {len(pairs)} sentence pairs")
    # Build vocabularies
    print("Building vocabularies...")
    src_vocab, tgt_vocab = build_vocabularies(pairs)
    print(f"English vocabulary size: {src_vocab.n_words}")
    print(f"Twi vocabulary size: {tgt_vocab.n_words}")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        pairs, 
        src_vocab, 
        tgt_vocab, 
        batch_size=BATCH_SIZE,
        train_split=0.9
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    print("Initializing model...")
    model = Transformer(
        src_vocab_size=src_vocab.n_words,
        tgt_vocab_size=tgt_vocab.n_words,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )

    device = setup_device()
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    criterion = create_loss_function()
    optimizer = create_optimizer(model, learning_rate=LEARNING_RATE)
    
    print("\nReady to train!")
    print("=" * 60)

    best_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS
    )
    
    print(f"\nTraining finished! Best validation loss: {best_loss:.4f}")
    print("Model saved as 'best_model.pt'")
if __name__ == "__main__":
    main()
    