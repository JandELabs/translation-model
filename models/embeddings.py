import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def split_heads(self, x):
        batch, seq_len, d_model = x.size()
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        batch, heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, heads * d_k)
    
    def forward(self, q, k, v, mask=None):
        Q = self.split_heads(self.W_q(q))
        K = self.split_heads(self.W_k(k))
        V = self.split_heads(self.W_v(v))
        
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = scores.softmax(dim=-1)
        context = weights @ V
        context = self.combine_heads(context)
        output = self.W_o(context)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
    
    def forward(self, x, mask=None):
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer2(x, lambda x: self.feed_forward(x))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1, max_len=5000):
        super(Encoder, self).__init__()
        self.embedding = Embeddings(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer2(x, lambda x: self.cross_attn(x, encoder_output, encoder_output, src_mask))
        x = self.sublayer3(x, lambda x: self.feed_forward(x))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1, max_len=5000):
        super(Decoder, self).__init__()
        self.embedding = Embeddings(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        x = self.norm(x)
        return self.output_projection(x)


def create_padding_mask(seq):
    """Creates padding mask where 0 indicates padding."""
    mask = (seq != 0).unsqueeze(1).unsqueeze(2)
    return mask.float()


def create_look_ahead_mask(size):
    """Creates lower triangular mask to prevent looking ahead."""
    mask = torch.tril(torch.ones(size, size))
    return mask.unsqueeze(0).unsqueeze(0)


def create_target_mask(tgt):
    """Creates combined padding + look-ahead mask for decoder."""
    tgt_seq_len = tgt.size(1)
    tgt_padding_mask = create_padding_mask(tgt)
    tgt_look_ahead_mask = create_look_ahead_mask(tgt_seq_len).to(tgt.device)
    tgt_mask = tgt_padding_mask & tgt_look_ahead_mask
    return tgt_mask.float()


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 d_ff=2048, num_layers=6, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout, max_len)
    
    def forward(self, src, tgt):
        """
        src: [batch_size, src_seq_len] - English word IDs
        tgt: [batch_size, tgt_seq_len] - Twi word IDs
        Returns: [batch_size, tgt_seq_len, tgt_vocab_size] - logits
        """
        src_mask = create_padding_mask(src)
        tgt_mask = create_target_mask(tgt)
        
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        return decoder_output
    
class Vocabulary:
    def __init__(self):
        # Special tokens - these MUST be first and in this order
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.word_count = {}
        self.n_words = 4  # Count of unique words (starts at 4 for special tokens)

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            # add new word
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.word_count[word] = 1
            self.n_words += 1
        else:
            # if word exists then increase count
            self.word_count[word] += 1
    
    def sentence_to_ids(self, sentence):
        ids = []
        for word in sentence.split():
            if word in self.word2idx:
                ids.append(self.word2idx[word])
            else:
                ids.append(self.word2idx['<UNK>'])
        return ids
    
    def ids_to_sentence(self, ids):
        words = []
        for idx in ids:
            if idx in self.idx2word:
                word = self.idx2word[idx]
                if word not in ['<PAD>', '<START>', '<END>']:
                    words.append(word)
        return ' '.join(words)
    
def load_parallel_data(filename):
    """Load parallel English-Twi data."""
    pairs = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # Remove whitespace
            if '|' in line:
                english, twi = line.split('|')
                pairs.append((english.strip(), twi.strip()))
    return pairs

def build_vocabularies(pairs):
    """Build English and Twi vocabularies from parallel data."""
    english_vocab = Vocabulary()
    twi_vocab = Vocabulary()
    
    for english_sent, twi_sent in pairs:
        english_vocab.add_sentence(english_sent.lower())
        twi_vocab.add_sentence(twi_sent.lower())
    
    return english_vocab, twi_vocab

class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        english_sent, twi_sent = self.pairs[idx]
        
        src_ids = self.src_vocab.sentence_to_ids(english_sent.lower())
        tgt_ids = self.tgt_vocab.sentence_to_ids(twi_sent.lower())
        
        tgt_input = [self.tgt_vocab.word2idx['<START>']] + tgt_ids
        tgt_output = tgt_ids + [self.tgt_vocab.word2idx['<END>']]
        
        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt_input': torch.tensor(tgt_input, dtype=torch.long),
            'tgt_output': torch.tensor(tgt_output, dtype=torch.long)
        }

def collate_fn(batch):
    """
    Pad sequences in a batch to same length.
    batch: List of dictionaries from __getitem__
    """
    # Separate the components
    src_batch = [item['src'] for item in batch]
    tgt_input_batch = [item['tgt_input'] for item in batch]
    tgt_output_batch = [item['tgt_output'] for item in batch]
    
    # Pad sequences
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_input_padded = torch.nn.utils.rnn.pad_sequence(tgt_input_batch, batch_first=True, padding_value=0)
    tgt_output_padded = torch.nn.utils.rnn.pad_sequence(tgt_output_batch, batch_first=True, padding_value=0)
    
    return {
        'src': src_padded,
        'tgt_input': tgt_input_padded,
        'tgt_output': tgt_output_padded
    }


def create_dataloaders(pairs, src_vocab, tgt_vocab, batch_size=32, train_split=0.9):
    """Create train and validation dataloaders."""
    
    # Split into train and validation
    split_idx = int(len(pairs) * train_split)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    # Create datasets
    train_dataset = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    val_dataset = TranslationDataset(val_pairs, src_vocab, tgt_vocab)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

