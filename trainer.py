import math
import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, optimizer, criterion, bptt, batch_size, num_epochs, model_name, max_grad_norm, early_stop_patience=5):
        self.optimizer = optimizer
        self.criterion = criterion
        self.bptt = bptt
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.model_name = model_name
        self.early_stop_patience = early_stop_patience
        
    def _get_batch(self, source, i):
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target

    def _detach_tensor(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach()
        else:
            return tuple(self._detach_tensor(t) for t in tensor)
    def _train_epoch(self, model, train_data):
        model.train()
        total_loss = 0.
        num_batches = 0
        hidden = None
        for batch, i in enumerate(range(0, train_data.size(0) - 1, self.bptt)):
            data, targets = self._get_batch(train_data, i)
            self.optimizer.zero_grad()
            
            hidden = None if hidden is None else self._detach_tensor(hidden) # Detach hidden state from the previous iteration
            output, hidden = model(data, hidden)
            loss = self.criterion(output.view(-1, model.fc.out_features), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            
        train_loss = total_loss / num_batches
        perplexity = math.exp(train_loss)
        return train_loss, perplexity

    def evaluate(self, model, data_source):
        model.eval()
        total_loss = 0.
        num_batches = 0
        hidden = None
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, self.bptt):
                data, targets = self._get_batch(data_source, i)
                output, hidden = model(data, hidden)
                output_flat = output.view(-1, model.fc.out_features)
                total_loss += len(data) * self.criterion(output_flat, targets).item()
                num_batches += 1
                
        eval_loss = total_loss / (len(data_source) - 1)
        perplexity = math.exp(eval_loss)
        return eval_loss, perplexity

    def train_model(self, model, train_data, valid_data, use_tb=True):
        best_valid_perplexity = float('inf')
        epochs_no_improve = 0
        checkpoint_dir = f'models/checkpoints/{self.model_name}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        if use_tb:
            try:
                input_dim = model.embedding.num_embeddings
                dummy_input = torch.randint(0, input_dim, (self.bptt, self.batch_size), dtype=torch.long).to(model.device)
                writer = SummaryWriter()
                writer.add_graph(model, dummy_input)
            except Exception as e:
                print(f"Failed to visualize model in TensorBoard: {e}")
                use_tb = False
        
        train_perplexities = []
        valid_perplexities = []
        epoch = 0
        for epoch in tqdm(range(self.num_epochs), desc="Training Progress"):
            train_loss, train_perplexity = self._train_epoch(model, train_data)
            valid_loss, valid_perplexity = self.evaluate(model, valid_data)
            
            train_perplexities.append(train_perplexity)
            valid_perplexities.append(valid_perplexity)
            
            tqdm.write(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Valid Loss: {valid_loss:.3f}, Train Perplexity: {train_perplexity:.3f}, Valid Perplexity: {valid_perplexity:.3f}')
            
            if valid_perplexity < best_valid_perplexity:
                best_valid_perplexity = valid_perplexity
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_validation_model.pt'))
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if use_tb:
                writer.add_scalars('Perplexity', {'Train': train_perplexity, 'Validation': valid_perplexity}, epoch)

            if epochs_no_improve >= self.early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
            
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_{epoch+1}.pt'))
        if use_tb:
            writer.close()
            
        return train_perplexities, valid_perplexities
