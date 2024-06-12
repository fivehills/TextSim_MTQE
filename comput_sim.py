
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
try:
    dataset = pd.read_csv('train_text.tsv', sep='\t', error_bad_lines=False)
except pd.errors.ParserError as e:
    print("Error reading the file:", e)
    exit()
##Skipping line 6172: expected 2 fields, saw 3
# Initialize the model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Function to compute cosine similarity between two sentences
def compute_similarity(row):
    sentences = [row['original'], row['translation']]
    embeddings = model.encode(sentences)
    similarity = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
    return similarity

# Apply the function to each row in the dataset
dataset['sen.sim'] = dataset.apply(compute_similarity, axis=1)

# Save the updated dataset
dataset.to_csv('text_similarity.tsv', sep='\t', index=False)

def compute_jaccard(rwo, text1, text2):
    text1 = [row['original']]
    text2 = row['translation']]
    set1, set2 = set(text1.split()), set(text2.split())
    return len(set1 & set2) / len(set1 | set2)
dataset['jaccard_sim'] = dataset.apply(compute_jaccard, axis=1)

# evaluate

def evaluate_translations(reference, translation):
    cosine_sim = compute_cosine_similarity(reference, translation)
    jaccard_idx = compute_jaccard_index(reference, translation)
    
    # Print or log the results
    print(f"Cosine Similarity: {cosine_sim}")
    print(f"Jaccard Index: {jaccard_idx}")



# tests/test_textual_similarity.py
import unittest

class TestTextualSimilarity(unittest.TestCase):
    def test_cosine_similarity(self):
        self.assertAlmostEqual(compute_cosine_similarity("hello world", "hello"), 0.5, places=2)
    
    def test_jaccard_index(self):
        self.assertAlmostEqual(compute_jaccard_index("hello world", "hello"), 1/3, places=2)

if __name__ == "__main__":
    unittest.main()

