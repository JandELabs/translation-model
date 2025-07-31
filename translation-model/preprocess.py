from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import torch

def read_file(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f]

def tokenize(sentence):
    return sentence.lower().strip().split()

def build_vocab(sentences):
    special_tokens = ["<pad>","<unk>","<sos>","<eos>"]
    counter = Counter()

    for sentence in sentences:
        tokens = tokenize(sentence)
        counter.update(tokens)
    
    vocab = {tok: idx for idx, tok in enumerate(special_tokens)}
    for word in counter:
        vocab[word] = len(vocab)
    return vocab

def numericalize(sentences,vocab):
    data = []
    for sentence in sentences:
        tokens = ["<sos>"] + tokenize(sentence) + ["<eos>"]
        numbers = [vocab.get(token,vocab["<unk>"]) for token in tokens]
        data.append(numbers)
    return data

def pad_sequences(sequences, pad_value=0):
    tensor_sequences = [torch.tensor(seq) for seq in sequences]
    return pad_sequence(tensor_sequences, batch_first=True,padding_value=pad_value)

if __name__ == "__main__":
    # en = read_file("data/train.en")
    # tw = read_file("data/train.tw")

    # print("First English sentence:", en[0])
    # print("First Twi sentence:", tw[0])

    en_sentences = read_file("data/train.en")
    tw_sentences = read_file("data/train.tw")

    en_vocab = build_vocab(en_sentences)
    tw_vocab = build_vocab(tw_sentences)

    # print("English vocab size:", len(en_vocab))
    # print("Twi vocab size:", len(tw_vocab))
    # print("Sample English tokens:", list(en_vocab.items()) [:10])

    en_numericalized = numericalize(en_sentences,en_vocab)
    tw_numericalized = numericalize(tw_sentences, tw_vocab)

    en_padded = pad_sequences(en_numericalized,pad_value=en_vocab["<pad>"])
    tw_padded = pad_sequences(tw_numericalized,pad_value=tw_vocab["<pad>"])

    print("Padded English shape:", en_padded.shape)
    print("First padded English sequences:", en_padded[0])
    print("Decoded:", [list(en_vocab.keys())[list(en_vocab.values()).index(i.item())] for i in en_padded[0]])

    # print("\nFirst English sentence:", en_sentences[0])
    # print("As numbers:", en_numericalized[0])
    
    # print("\nFirst Twi sentence:", tw_sentences[0])
    # print("As numbers:", tw_numericalized[0])
