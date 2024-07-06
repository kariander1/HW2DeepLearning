import torch
import torch.nn as nn
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, device, dropout, model_type='LSTM', checkpoint = None):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.model_type = model_type
        self.output_dim = output_dim
        
        if model_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout if n_layers > 1 else 0)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout if n_layers > 1 else 0)
        else:
            raise ValueError("Invalid model type. Choose from 'LSTM' or 'GRU'.")

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        
        
        if checkpoint:
            checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
            self.load_state_dict(checkpoint)
            print(f"Loaded model from checkpoint: {checkpoint}")
        self.to(device)
            
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.rnn(embedded)
        output = self.fc(self.dropout(output))
        return output
