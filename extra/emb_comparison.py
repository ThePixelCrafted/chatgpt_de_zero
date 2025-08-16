"""
This script demonstrates how to load a word2vec model and find the closest words to given vectors.
It uses the Google News word2vec model, which is a pre-trained model available in the gensim library.
We can see funny examples of word vector arithmetic, such as "king - man + woman = queen".
"""

from gensim.models import KeyedVectors
import kagglehub

# pip install -U gensim. Careful, the numpy version will be downgraded to <2

# Download latest version
_dir = kagglehub.dataset_download("sugataghosh/google-word2vec")
path_model = _dir + "/GoogleNews-vectors-negative300.bin"

# Load the model
print("Loading the model...")
model = KeyedVectors.load_word2vec_format(path_model, binary=True)
print("Model loaded successfully.")


def print_closest_words(vector, topn=5):
    """Get the closest words to a given vector."""
    # Get the top N most similar words
    similar_words = model.similar_by_vector(vector, topn=topn)
    for word, similarity in similar_words:
        print(f"{word}: {similarity:.4f}")


print("Closest words to 'actor - man + woman':")
print_closest_words(model["actor"] - model["man"] + model["woman"])

print("\n\nClosest words to 'king - man + woman':")
print_closest_words(model["king"] - model["man"] + model["woman"])
