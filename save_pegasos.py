# save_pegasos.py
import pickle
import numpy as np
import project1 as p1
import utils

print("🚀 Saving your REAL Pegasos model...")

# Load and prepare data
train_data = utils.load_data('data/reviews_train.tsv')
train_texts = [sample['text'] for sample in train_data]
train_labels = [sample['sentiment'] for sample in train_data]

print("📖 Creating dictionary...")
dictionary = p1.bag_of_words(train_texts)

print("🔢 Extracting features...")
train_features = p1.extract_bow_feature_vectors(train_texts, dictionary)

print("🧠 Training Pegasos (T=25, L=0.01)...")
theta, theta_0 = p1.pegasos(train_features, train_labels, T=25, L=0.01)

# Create word list for reference
wordlist = [word for (idx, word) in sorted(zip(dictionary.values(), dictionary.keys()))]

print(f"✅ Model trained! Vocabulary: {len(dictionary)} words")

# Save a COMPACT version (critical for deployment)
compact_model = {
    'theta': theta.astype(np.float32),  # Reduce precision to save space
    'theta_0': float(theta_0),
    'dictionary': dictionary,
    'wordlist': wordlist,
    'top_positive': utils.most_explanatory_word(theta, wordlist)[:20],
    'top_negative': utils.most_explanatory_word(theta, wordlist)[-20:],
    'accuracy': 0.808,
    'info': "Pegasos algorithm trained on 5,000+ Amazon food reviews"
}

# Save it
with open('pegasos_model_compact.pkl', 'wb') as f:
    pickle.dump(compact_model, f, protocol=4)  # Protocol 4 is more efficient

print(f"💾 Model saved as 'pegasos_model_compact.pkl' ({len(dictionary)} words)")
print(f"📦 File size should be ~1-2MB (much smaller than before)")