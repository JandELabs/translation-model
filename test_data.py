from models.embeddings import load_parallel_data, build_vocabularies

# Test loading
print("Testing data loading...")
pairs = load_parallel_data('data/parallel_data.txt')
print(f"✓ Loaded {len(pairs)} pairs")

# Show first 3 pairs
print("\nFirst 3 pairs:")
for i, (eng, twi) in enumerate(pairs[:3]):
    print(f"{i+1}. {eng} | {twi}")

# Test vocabulary building
print("\nBuilding vocabularies...")
src_vocab, tgt_vocab = build_vocabularies(pairs)
print(f"✓ English vocab: {src_vocab.n_words} words")
print(f"✓ Twi vocab: {tgt_vocab.n_words} words")

print("\n✓ Everything works!")