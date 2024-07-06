import os
import torch
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter

tokenizer = get_tokenizer('basic_english')

# Section 1: Load and Visualize the Penn Tree Bank Dataset
def _load_data(file_path):
    with open(file_path, 'r') as f:
        return f.read().split()
    
def _yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

def _data_process(raw_text_iter, vocab):
    data = [torch.tensor([vocab[token] for token in tokenizer(item)], dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

def _create_batches(data, batch_size):
    nbatch = data.size(0) // batch_size
    data = data.narrow(0, 0, nbatch * batch_size)
    data = data.view(batch_size, -1).t().contiguous()
    return data

def create_penn_tree_bank_dataset(data_dir, batch_size, device):
    # Load the word level Penn Tree Bank dataset. For character level, use 'ptb.char.train.txt' etc.
    train_file = os.path.join(data_dir, 'ptb.train.txt')
    valid_file = os.path.join(data_dir, 'ptb.valid.txt')
    test_file = os.path.join(data_dir, 'ptb.test.txt')

    train_data = _load_data(train_file)
    valid_data = _load_data(valid_file)
    test_data = _load_data(test_file)

    vocab = build_vocab_from_iterator(_yield_tokens(train_data), specials=['<unk>', '<pad>', '<eos>'])
    vocab.set_default_index(vocab['<unk>'])
    
    train_data = _data_process(train_data, vocab).to(device)
    valid_data = _data_process(valid_data, vocab).to(device)
    test_data = _data_process(test_data, vocab).to(device)

    train_data = _create_batches(train_data, batch_size)
    valid_data = _create_batches(valid_data, batch_size)
    test_data = _create_batches(test_data, batch_size)
    
    return train_data, valid_data, test_data, vocab